"""Batimetria e Equações de Maré de Laplace.

Os testes caros rodam a 4 graus, resolução em que o modelo global roda em
segundos e ainda assim produz anfidromos -- o fenômeno que se quer verificar é
qualitativo (existe? gira para que lado?), não quantitativo.
"""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.bathymetry import Bathymetry, coarsen_to_grid
from oceantides.constants import G_SURFACE, OMEGA_EARTH, R_EARTH
from oceantides.harmonic import CONSTITUENTS
from oceantides.lte import LaplaceTidalModel, LTEConfig

pytest.importorskip("xarray", reason="batimetria requer o extra 'data'")


@pytest.fixture(scope="module")
def bathy():
    return Bathymetry(resolution=4.0)


@pytest.fixture(scope="module")
def solution():
    """Solução M2 grosseira, reaproveitada por vários testes."""
    cfg = LTEConfig(
        resolution=4.0, days=9.0, spin_up_days=7.0, forcing="constituent",
        constituent="M2", internal_drag=1e-5, sample_every=6,
    )
    return LaplaceTidalModel(cfg).run(constituents=["M2"])


class TestCoarsen:
    def test_averages_only_ocean_subcells(self):
        """Misturar cotas de terra na média rebaixaria células costeiras."""
        block = np.array([[-100.0, -200.0], [50.0, 80.0]])  # metade terra
        depth, frac = coarsen_to_grid(block, 2)
        assert depth[0, 0] == pytest.approx(150.0)  # média de 100 e 200, só
        assert frac[0, 0] == pytest.approx(0.5)

    def test_all_land_gives_zero_depth(self):
        depth, frac = coarsen_to_grid(np.full((2, 2), 300.0), 2)
        assert depth[0, 0] == 0.0 and frac[0, 0] == 0.0

    def test_rejects_indivisible_grid(self):
        with pytest.raises(ValueError):
            coarsen_to_grid(np.zeros((7, 7)), 2)


class TestBathymetry:
    def test_ocean_covers_about_two_thirds(self, bathy):
        """~71% do globo é oceano; aqui dá menos porque acima de 86 graus tudo
        vira terra e células com menos de metade de água são descartadas."""
        assert 0.6 < bathy.mask.mean() < 0.72

    def test_mean_depth_is_realistic(self, bathy):
        d = bathy.depth[bathy.mask]
        assert 3000 < d.mean() < 3900
        assert 3500 < np.median(d) < 4100

    def test_min_depth_is_enforced(self, bathy):
        assert bathy.depth[bathy.mask].min() >= bathy.min_depth

    def test_polar_cap_is_masked(self, bathy):
        beyond = np.abs(bathy.lat) > bathy.lat_limit
        assert not bathy.mask[beyond].any()

    def test_wave_speed_matches_shallow_water(self, bathy):
        c = bathy.wave_speed()[bathy.mask]
        h = bathy.depth[bathy.mask]
        np.testing.assert_allclose(c, np.sqrt(G_SURFACE * h))

    def test_deep_ocean_wave_speed_is_about_200(self, bathy):
        """c = sqrt(g*4000) ~ 198 m/s: o número que explica tudo."""
        assert np.sqrt(G_SURFACE * 4000.0) == pytest.approx(198.0, abs=2.0)

    def test_wave_is_slower_than_the_sublunar_point(self, bathy):
        """A onda perde a corrida para a Lua -- por isso não há bojo seguidor.

        A velocidade a comparar é a do ponto sublunar **relativa ao solo**,
        ``2 pi R / dia lunar`` ~ 448 m/s, e não ``Omega R`` ~ 465 m/s: esta
        última é a velocidade de um ponto do equador no espaço inercial, que
        não é com quem a onda compete.
        """
        from oceantides.constants import T_LUNAR_DAY

        c = float(np.sqrt(G_SURFACE * np.median(bathy.depth[bathy.mask])))
        sublunar = 2.0 * np.pi * R_EARTH / T_LUNAR_DAY
        assert sublunar == pytest.approx(448.0, abs=3.0)
        assert sublunar / c > 2.0

    def test_cfl_shrinks_with_resolution(self):
        coarse = Bathymetry(resolution=4.0).cfl_timestep()
        fine = Bathymetry(resolution=2.0).cfl_timestep()
        assert fine < coarse
        assert fine / coarse == pytest.approx(0.5, rel=0.25)

    def test_rejects_non_multiple_resolution(self):
        with pytest.raises(ValueError):
            Bathymetry(resolution=0.3)


class TestForcing:
    def test_constituent_mode_matches_the_analytic_form(self):
        """A forçante monocromática deve ser A f(phi) cos(sigma t + n lambda)."""
        from oceantides.constants import LOVE_FACTOR

        m = LaplaceTidalModel(
            LTEConfig(resolution=4.0, forcing="constituent", constituent="M2")
        )
        t = 12345.0
        field = m.equilibrium_field(t)

        con = CONSTITUENTS["M2"]
        phi = np.radians(m.lat)[:, None]
        lam = np.radians(m.lon)[None, :]
        expected = (
            con.amplitude * LOVE_FACTOR * np.cos(phi) ** 2
            * np.cos(con.omega * t + 2 * lam)
        )
        np.testing.assert_allclose(field, expected, atol=1e-12)

    def test_diurnal_forcing_vanishes_at_the_equator(self):
        """A forma das diurnas é sin(2 phi), nula no equador.

        Não é um defeito: é a razão física de as marés diurnas serem fracas em
        latitudes baixas.
        """
        m = LaplaceTidalModel(
            LTEConfig(resolution=4.0, forcing="constituent", constituent="K1")
        )
        j = int(np.argmin(np.abs(m.lat)))
        assert np.abs(m.equilibrium_field(5000.0)[j]).max() < 0.02

    def test_astronomical_forcing_has_two_bulges(self):
        m = LaplaceTidalModel(LTEConfig(resolution=4.0, forcing="astronomical"))
        field = m.equilibrium_field(0.0)
        assert np.nanmax(field) > 0 and np.nanmin(field) < 0


class TestConservation:
    def test_volume_is_conserved(self):
        """A continuidade está em forma de fluxo, então o volume total é
        conservado exatamente -- a forçante entra só no momento.

        É o teste mais forte da discretização: qualquer erro de indexação nos
        fluxos das faces quebra isso imediatamente.
        """
        m = LaplaceTidalModel(
            LTEConfig(resolution=4.0, forcing="constituent", constituent="M2")
        )
        area = np.cos(np.radians(m.lat))[:, None] * m.mask

        state = (np.zeros(m.mask.shape), np.zeros(m.mask.shape), np.zeros(m.mask.shape))
        volumes = []
        for k in range(400):
            state = m.step(state, k * m.dt, m.dt)
            volumes.append(float(np.sum(state[0] * area)))

        volumes = np.array(volumes)
        scale = np.abs(state[0]).max() * area.sum()
        assert np.abs(volumes).max() < 1e-9 * max(scale, 1e-12)

    def test_stays_finite(self):
        m = LaplaceTidalModel(
            LTEConfig(resolution=4.0, forcing="constituent", constituent="M2")
        )
        state = (np.zeros(m.mask.shape), np.zeros(m.mask.shape), np.zeros(m.mask.shape))
        for k in range(2000):
            state = m.step(state, k * m.dt, m.dt)
        assert np.isfinite(state[0]).all()
        assert np.abs(state[0]).max() < 50.0  # nada de explosão numérica

    def test_no_flow_through_coasts(self):
        m = LaplaceTidalModel(
            LTEConfig(resolution=4.0, forcing="constituent", constituent="M2")
        )
        state = (np.zeros(m.mask.shape), np.zeros(m.mask.shape), np.zeros(m.mask.shape))
        for k in range(200):
            state = m.step(state, k * m.dt, m.dt)
        assert np.all(state[1][~m.mask_u] == 0.0)
        assert np.all(state[2][~m.mask_v] == 0.0)
        assert np.all(state[0][~m.mask] == 0.0)


class TestAmphidromes:
    def test_detects_a_synthetic_winding_point(self, solution):
        """Campo de fase sintético girando 360 graus em torno de um ponto."""
        lat, lon = solution.lat, solution.lon
        j0, i0 = len(lat) // 2, len(lon) // 2
        dj = np.arange(len(lat))[:, None] - j0
        di = np.arange(len(lon))[None, :] - i0
        phase = np.degrees(np.arctan2(dj, di)) % 360.0
        amp = np.hypot(dj, di) * 0.01

        import copy

        probe = copy.copy(solution)
        probe.cos_coeff = np.array([amp * np.cos(np.radians(phase))])
        probe.sin_coeff = np.array([amp * np.sin(np.radians(phase))])
        probe.mask = np.ones_like(solution.mask)
        probe.depth = np.full_like(solution.depth, 4000.0)

        found = probe.find_amphidromes("M2", amplitude_threshold=0.02)
        assert len(found) == 1
        assert found[0][0] == pytest.approx(lat[j0], abs=abs(lat[1] - lat[0]))

    def test_model_produces_amphidromes(self, solution):
        """A prova de que o oceano responde como onda, e não como bojo."""
        found = solution.find_amphidromes("M2", 0.03)
        assert len(found) >= 6, f"apenas {len(found)} anfidromos"

    def test_rotation_follows_coriolis(self, solution):
        """Anti-horário no norte, horário no sul -- assinatura de Coriolis.

        Não é imposto em lugar nenhum do código: emerge do termo ``f = 2 Omega
        sin(phi)`` nas equações do momento.
        """
        stats = solution.amphidrome_rotation_stats("M2", amplitude_threshold=0.03)
        for hemisphere, s in stats.items():
            if s["total"] >= 4:
                assert s["fracao_esperada"] > 0.6, (hemisphere, s)

    def test_equilibrium_theory_has_no_amphidromes(self):
        """Contraprova: a maré de equilíbrio não tem onde ter anfidromo.

        Ela está sempre em fase com o potencial, então a fase só assume dois
        valores (0 e 180 graus) e nunca circula.
        """
        from oceantides.viz.tide_wave import equilibrium_constituent_field

        lat = np.linspace(-88, 88, 45)
        lon = np.linspace(-180, 176, 90)
        field = equilibrium_constituent_field(lat, lon, "M2", 0.0)
        phase = np.degrees(np.arctan2(0.0 * field, field)) % 360.0
        assert len(np.unique(np.round(phase))) <= 2


class TestSolutionAmplitude:
    def test_deep_ocean_amplitude_is_realistic(self, solution):
        """M2 em oceano profundo: ~0.2-0.5 m nos modelos globais reais."""
        stats = solution.global_stats("M2")
        assert 0.1 < stats["amplitude_media_oceano_profundo"] < 0.6

    def test_maximum_is_on_a_shelf_not_in_the_deep(self, solution):
        """As maiores amplitudes ficam em água rasa, onde a onda desacelera e
        se amplifica -- lei de Green."""
        amp, _ = solution.constituent("M2")
        j, i = np.unravel_index(np.nanargmax(amp), amp.shape)
        assert solution.depth[j, i] < np.median(solution.depth[solution.mask])

    def test_currents_are_available_and_bounded(self, solution):
        speed = solution.current_amplitude("M2")
        assert np.nanmax(speed) < 8.0
        assert np.nanmean(speed) > 0.0

    def test_water_flows_where_the_tide_does_not_rise(self, solution):
        """No anfidromo a maré não sobe, mas a corrente é normal.

        Cuidado com a formulação: elevação e corrente estão em quadratura de
        **fase**, não de amplitude. Espacialmente as duas amplitudes são até
        bem correlacionadas (~0.8), porque plataforma rasa tem maré grande *e*
        corrente grande. A afirmação correta e verificável é pontual:

        * elevação no anfidromo ~ 6% da mediana global (praticamente zero);
        * corrente no anfidromo ~ 100% da mediana global (nada de especial).

        A energia atravessa o ponto; o que não acontece ali é a maré subir.
        """
        amp, _ = solution.constituent("M2")
        speed = solution.current_amplitude("M2")
        found = solution.find_amphidromes("M2", 0.03)
        assert len(found) >= 5

        js = [int(np.argmin(np.abs(solution.lat - a[0]))) for a in found]
        iss = [int(np.argmin(np.abs(solution.lon - a[1]))) for a in found]
        elev = np.median([amp[j, i] for j, i in zip(js, iss)])
        curr = np.median([speed[j, i] for j, i in zip(js, iss)])

        assert elev / np.nanmedian(amp) < 0.25  # elevação some
        assert curr / np.nanmedian(speed) > 0.5  # corrente não
