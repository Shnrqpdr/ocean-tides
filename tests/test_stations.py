"""Comportamento das estações -- e onde o modelo de 1 GL deixa de valer.

Estes testes registram o desempenho *medido* do modelo contra amplitudes
observadas, inclusive a discrepância conhecida em bacias ressonantes de alta
latitude. Se alguém melhorar o modelo, é aqui que o ganho aparece.
"""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.constants import T_M2
from oceantides.response import steady_state_response
from oceantides.simulation import SimConfig, run_simulation
from oceantides.stations import DEFAULT_STATIONS, get_stations


@pytest.fixture(scope="module")
def result():
    return run_simulation(SimConfig(days=30.0, dt=120.0, sample_dt=900.0))


def _range(result, name):
    dyn, _ = result.station(name)
    return float(np.ptp(dyn))


class TestLookup:
    def test_all_by_default(self):
        assert len(get_stations()) == len(DEFAULT_STATIONS)

    def test_by_name_is_case_insensitive(self):
        assert get_stations(["rio de janeiro"])[0].name == "Rio de Janeiro"

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            get_stations(["Atlantis"])

    def test_positions_are_on_the_sphere(self):
        from oceantides.constants import R_EARTH
        from oceantides.stations import station_positions

        radii = np.linalg.norm(station_positions(DEFAULT_STATIONS), axis=-1)
        np.testing.assert_allclose(radii, R_EARTH, rtol=1e-12)


class TestLowLatitudeAccuracy:
    """Onde o modelo acerta: bacias rasas e de baixa latitude."""

    @pytest.mark.parametrize(
        "name,observed,tol",
        [
            ("Rio de Janeiro", 1.2, 0.4),
            ("Santos", 1.4, 0.5),
            ("Belem", 3.5, 1.0),
            ("Salvador", 2.3, 1.0),
        ],
    )
    def test_within_tolerance_of_observed(self, result, name, observed, tol):
        assert _range(result, name) == pytest.approx(observed, abs=tol)


class TestKnownModelLimit:
    """Onde o modelo NÃO acerta -- documentado, não escondido."""

    def test_fundy_is_underpredicted_by_about_threefold(self, result):
        """Fundy dá ~5 m contra 16 m observados.

        Não é erro de ajuste: a maré de equilíbrio *local* em 45 graus vale só
        ~66% da equatorial, e os 16 m reais vêm da co-oscilação com o sistema
        anfidrômico do Atlântico Norte -- um modo de bacia oceânica que um
        oscilador de um grau de liberdade forçado localmente não representa.
        """
        modelled = _range(result, "Burntcoat Head")
        assert 3.0 < modelled < 9.0
        assert modelled < 16.0 / 1.8

    def test_high_latitude_equilibrium_forcing_is_weaker(self, result):
        """A causa física da subestimação, medida diretamente."""
        _, eq_fundy = result.station("Burntcoat Head")
        _, eq_rio = result.station("Rio de Janeiro")
        ratio = np.ptp(eq_fundy) / np.ptp(eq_rio)
        assert ratio == pytest.approx(0.70, abs=0.12)


class TestResponseContrast:
    """O contraste qualitativo que o PDF descreve na p. 203."""

    def test_ordering_follows_observation(self, result):
        modelled = [_range(result, s.name) for s in DEFAULT_STATIONS]
        observed = [s.observed_range for s in DEFAULT_STATIONS]
        # correlação de postos entre modelado e observado
        rank = lambda v: np.argsort(np.argsort(v))
        corr = np.corrcoef(rank(modelled), rank(observed))[0, 1]
        assert corr > 0.85, f"correlação de postos {corr:.2f}"

    def test_resonant_basin_beats_rigid_basin(self, result):
        assert _range(result, "Burntcoat Head") > 3 * _range(result, "Rio de Janeiro")

    def test_sluggish_basin_responds_below_equilibrium(self, result):
        """Taiti: T0 > T_M2 -> a bacia não acompanha, e A < 1.

        É o caso-limite oposto ao de Fundy, e o modelo o captura corretamente.
        """
        dyn, eq = result.station("Papeete")
        assert np.ptp(dyn) < np.ptp(eq)

    def test_resonant_basins_lag_more(self, result):
        """Quanto mais perto da ressonância, maior o atraso da maré alta."""
        lags = dict(zip(result.station_names, result.meta["m2_lag_hours"]))
        assert lags["Burntcoat Head"] > lags["Rio de Janeiro"] + 2.0

    def test_amplification_matches_analytic_admittance(self, result):
        for i, station in enumerate(get_stations(result.config.stations)):
            amp, _ = steady_state_response(
                2 * np.pi / station.natural_period, station.q_factor, 2 * np.pi / T_M2
            )
            assert result.meta["m2_amplification"][i] == pytest.approx(
                float(amp), rel=1e-12
            )


class TestSpinUp:
    def test_output_starts_in_steady_state(self, result):
        """Sem spin-up a amplitude sobe por semanas e mascara a sizígia.

        Compara a amplitude da primeira sizígia com a da terceira: em regime
        permanente elas devem ser praticamente iguais.
        """
        dyn, _ = result.station("Burntcoat Head")
        day = result.days
        first = np.ptp(dyn[day < 2.0])
        third = np.ptp(dyn[(day > 28.0)])
        assert first == pytest.approx(third, rel=0.20)

    def test_disabling_spin_up_shows_the_transient(self):
        """Confirma que o spin-up é mesmo o que corrige o problema."""
        raw = run_simulation(
            SimConfig(days=30.0, dt=300.0, sample_dt=1800.0, spin_up=False,
                      stations=("Burntcoat Head",))
        )
        dyn = raw.h_dynamic[:, 0]
        early = np.ptp(dyn[raw.days < 2.0])
        late = np.ptp(dyn[raw.days > 28.0])
        assert late > 1.5 * early  # ainda subindo ao regime
