"""Reproduz os números impressos no Exemplo 5.5 e no texto das pp. 202-204."""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.constants import (
    A_EARTH_ORBIT,
    A_MOON,
    BOOK,
    GM_EARTH,
    GM_MOON,
    GM_SUN,
    R_EARTH,
)
from oceantides.forcing import equilibrium_height, tidal_range

# Constantes exatas do livro, para bater com os valores impressos
GM_MOON_BOOK = BOOK.G * BOOK.M_MOON
GM_SUN_BOOK = BOOK.G * BOOK.M_SUN

BOOK_KW = dict(radius=BOOK.R_EARTH, g=BOOK.G_SURFACE)


class TestExample55:
    """Exemplo 5.5, p. 203: h = 3 G M_m r^2 / (2 g D^3) = 0.54 m."""

    def test_lunar_range_is_054_m(self):
        h = tidal_range(BOOK.D_MOON, GM_MOON_BOOK, **BOOK_KW)
        assert h == pytest.approx(0.54, abs=0.005)
        assert h == pytest.approx(0.5377, rel=1e-3)  # trava o valor exato

    def test_solar_range(self):
        h = tidal_range(BOOK.D_SUN, GM_SUN_BOOK, **BOOK_KW)
        assert h == pytest.approx(0.2461, rel=1e-3)

    def test_sun_tidal_effect_is_046_of_moon(self):
        """p. 202: 'the tidal force due to the Sun is 0.46 that of the Moon'."""
        ratio = tidal_range(BOOK.D_SUN, GM_SUN_BOOK, **BOOK_KW) / tidal_range(
            BOOK.D_MOON, GM_MOON_BOOK, **BOOK_KW
        )
        assert ratio == pytest.approx(0.46, abs=0.005)

    def test_sun_direct_attraction_is_about_175x(self):
        """p. 202: a atração direta do Sol é ~175x a da Lua na superfície.

        O contraste com o teste anterior é o ponto central da seção: o Sol
        puxa 175x mais forte, mas produz menos da metade da maré, porque o que
        importa é o *gradiente* do campo, que cai com 1/D^3 em vez de 1/D^2.
        """
        ratio = (BOOK.M_SUN / BOOK.D_SUN**2) / (BOOK.M_MOON / BOOK.D_MOON**2)
        assert ratio == pytest.approx(175, rel=0.05)

    def test_r_over_D_small_parameter(self):
        """p. 200: 'because r/D = 0.02'."""
        assert BOOK.R_EARTH / BOOK.D_MOON == pytest.approx(0.02, abs=0.004)


class TestSpringAndNeap:
    """p. 203: sizígia e quadratura."""

    @staticmethod
    def _h_moon():
        return tidal_range(BOOK.D_MOON, GM_MOON_BOOK, **BOOK_KW)

    @staticmethod
    def _h_sun():
        return tidal_range(BOOK.D_SUN, GM_SUN_BOOK, **BOOK_KW)

    def test_spring_tide_factor_is_146(self):
        """Alinhados: as duas marés somam -> (1 + 0.46) h = 1.46 h."""
        factor = 1.0 + self._h_sun() / self._h_moon()
        assert factor == pytest.approx(1.46, abs=0.01)

    def test_spring_tide_height(self):
        """ERRATUM DO LIVRO.

        O texto diz "should be 1.46h = 0.83 m", mas com h = 0.54 m o produto
        é 1.46 x 0.54 = 0.79 m, não 0.83 m. O fator e o valor impressos são
        mutuamente inconsistentes (0.83/1.46 = 0.57 m, que não é o h obtido no
        próprio Exemplo 5.5). Travamos aqui o valor aritmeticamente correto.
        """
        spring = self._h_moon() + self._h_sun()
        assert spring == pytest.approx(0.785, rel=1e-2)
        assert spring == pytest.approx(1.46 * self._h_moon(), rel=1e-2)
        assert spring != pytest.approx(0.83, abs=0.01)  # o valor impresso não fecha

    def test_neap_tide_height(self):
        """Em quadratura os efeitos se cancelam parcialmente: (1 - 0.46) h."""
        neap = self._h_moon() - self._h_sun()
        assert neap == pytest.approx(0.292, rel=1e-2)

    def test_spring_over_neap_ratio(self):
        assert (self._h_moon() + self._h_sun()) / (
            self._h_moon() - self._h_sun()
        ) == pytest.approx(2.69, rel=1e-2)


class TestEquilibriumSurface:
    """h_eq(psi), derivada do potencial, deve recair na Eq. 5.55."""

    @staticmethod
    def _point(psi_deg):
        p = np.radians(psi_deg)
        return np.array([R_EARTH * np.cos(p), R_EARTH * np.sin(p), 0.0])

    def test_high_minus_low_equals_eq_555(self):
        moon = np.array([A_MOON, 0.0, 0.0])
        high = equilibrium_height(self._point(0), moon, GM_MOON)
        low = equilibrium_height(self._point(90), moon, GM_MOON)
        assert high - low == pytest.approx(tidal_range(A_MOON, GM_MOON), rel=1e-12)

    def test_two_bulges_per_revolution(self):
        """A Terra gira sob dois bojos -> duas marés altas por dia (p. 202)."""
        moon = np.array([A_MOON, 0.0, 0.0])
        psi = np.linspace(0, 360, 721)
        h = np.array([equilibrium_height(self._point(a), moon, GM_MOON) for a in psi])
        # h ~ (3cos^2 psi - 1): máximos em 0 e 180, mínimos em 90 e 270
        assert h[0] == pytest.approx(h[360], rel=1e-12)  # 0 deg vs 180 deg
        assert h[180] == pytest.approx(h[540], rel=1e-12)  # 90 deg vs 270 deg
        assert h[0] > h[180]

    def test_mean_over_sphere_is_zero(self):
        """<3cos^2 psi - 1> = 0 sobre a esfera: a maré não adiciona água."""
        moon = np.array([A_MOON, 0.0, 0.0])
        # amostragem uniforme em área: cos(colatitude) uniforme
        rng = np.random.default_rng(0)
        u = rng.uniform(-1, 1, 200_000)
        phi = rng.uniform(0, 2 * np.pi, 200_000)
        s = np.sqrt(1 - u**2)
        pts = R_EARTH * np.stack([s * np.cos(phi), s * np.sin(phi), u], axis=-1)
        h = equilibrium_height(pts, moon, GM_MOON)
        assert abs(h.mean()) < 0.01 * h.std()

    def test_far_side_bulge_equals_near_side(self):
        """O bojo antípoda tem a mesma altura -- o resultado contraintuitivo
        que Galileu não conseguiu explicar (p. 198)."""
        moon = np.array([A_MOON, 0.0, 0.0])
        near = equilibrium_height(self._point(0), moon, GM_MOON)
        far = equilibrium_height(self._point(180), moon, GM_MOON)
        assert near == pytest.approx(far, rel=1e-12)

    def test_love_factor_reduces_range(self):
        moon = np.array([A_MOON, 0.0, 0.0])
        rigid = equilibrium_height(self._point(0), moon, GM_MOON)
        elastic = equilibrium_height(self._point(0), moon, GM_MOON, love=True)
        assert elastic / rigid == pytest.approx(0.69, abs=0.001)


class TestModernConstants:
    """Com as constantes atuais os valores mudam pouco -- sanidade."""

    def test_lunar_range_close_to_book(self):
        assert tidal_range(A_MOON, GM_MOON) == pytest.approx(0.54, abs=0.02)

    def test_solar_ratio_close_to_book(self):
        ratio = tidal_range(A_EARTH_ORBIT, GM_SUN) / tidal_range(A_MOON, GM_MOON)
        assert ratio == pytest.approx(0.46, abs=0.02)

    def test_surface_gravity_consistent(self):
        assert GM_EARTH / R_EARTH**2 == pytest.approx(9.82, abs=0.05)
