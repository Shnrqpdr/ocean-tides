"""Verifica forcing.py contra as Eq. 5.51-5.54 do PDF."""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.constants import GM_MOON, R_EARTH, A_MOON
from oceantides.forcing import (
    tidal_accel_exact,
    tidal_accel_expanded,
    tidal_potential,
)

# Lua sobre o eixo +x, a distância média
MOON = np.array([A_MOON, 0.0, 0.0])


def surface_point(theta_deg, radius=R_EARTH):
    """Ponto na superfície a um ângulo theta do eixo Terra-Lua (plano xy)."""
    t = np.radians(theta_deg)
    return np.array([radius * np.cos(t), radius * np.sin(t), 0.0])


class TestExpandedMatchesBook:
    """As Eq. 5.52/5.53/5.54 escritas à mão devem sair da forma vetorial."""

    def test_eq_5_52_sublunar_point(self):
        # Ponto b da Fig. 5-10b: theta = 0, F_Tx = +2 G M r / D^3
        a = tidal_accel_expanded(surface_point(0), MOON, GM_MOON)
        expected = 2.0 * GM_MOON * R_EARTH / A_MOON**3
        assert a[0] == pytest.approx(expected, rel=1e-12)
        assert a[1] == pytest.approx(0.0, abs=1e-20)

    def test_eq_5_52_antipodal_point(self):
        # Ponto a: theta = 180. Mesma magnitude, sentido oposto -- ambos
        # apontam para FORA do centro da Terra (os dois bojos).
        a = tidal_accel_expanded(surface_point(180), MOON, GM_MOON)
        expected = 2.0 * GM_MOON * R_EARTH / A_MOON**3
        assert a[0] == pytest.approx(-expected, rel=1e-12)

    def test_eq_5_53_quadrature_point(self):
        # Ponto c: theta = 90, F_Ty = -G M r / D^3 (para dentro).
        a = tidal_accel_expanded(surface_point(90), MOON, GM_MOON)
        expected = -GM_MOON * R_EARTH / A_MOON**3
        assert a[1] == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("theta", [0, 30, 45, 60, 90, 120, 180, 270])
    def test_eq_5_54_components(self, theta):
        """Eq. 5.54a/b: F_Tx = 2GMr cos(t)/D^3, F_Ty = -GMr sin(t)/D^3."""
        a = tidal_accel_expanded(surface_point(theta), MOON, GM_MOON)
        t = np.radians(theta)
        k = GM_MOON * R_EARTH / A_MOON**3
        assert a[0] == pytest.approx(2.0 * k * np.cos(t), abs=1e-16)
        assert a[1] == pytest.approx(-k * np.sin(t), abs=1e-16)


class TestBulgeGeometry:
    """A Fig. 5-11a: para fora nos polos do eixo Terra-Lua, para dentro em 90."""

    @pytest.mark.parametrize("theta", [0, 180])
    def test_outward_along_moon_axis(self, theta):
        p = surface_point(theta)
        a = tidal_accel_exact(p, MOON, GM_MOON)
        assert np.dot(a, p / np.linalg.norm(p)) > 0  # componente radial positiva

    @pytest.mark.parametrize("theta", [90, 270])
    def test_inward_at_quadrature(self, theta):
        p = surface_point(theta)
        a = tidal_accel_exact(p, MOON, GM_MOON)
        assert np.dot(a, p / np.linalg.norm(p)) < 0


class TestExactVersusExpanded:
    """O erro da expansão é O(r/D) ~ 2%, como o PDF afirma na p. 200."""

    @pytest.mark.parametrize("theta", [0, 30, 45, 60, 90, 120, 180])
    def test_within_two_and_a_half_percent(self, theta):
        p = surface_point(theta)
        exact = tidal_accel_exact(p, MOON, GM_MOON)
        approx = tidal_accel_expanded(p, MOON, GM_MOON)
        rel = np.linalg.norm(exact - approx) / np.linalg.norm(exact)
        assert rel < 0.025, f"theta={theta}: erro {rel:.3%}"

    def test_error_scales_linearly_with_r_over_D(self):
        """Dobrar r/D deve dobrar o erro -- confirma que a expansão é O(r/D)."""
        errs = []
        for radius in (R_EARTH, 2 * R_EARTH):
            p = surface_point(0, radius=radius)
            exact = tidal_accel_exact(p, MOON, GM_MOON)
            approx = tidal_accel_expanded(p, MOON, GM_MOON)
            errs.append(np.linalg.norm(exact - approx) / np.linalg.norm(exact))
        assert errs[1] / errs[0] == pytest.approx(2.0, rel=0.05)


class TestPotentialConsistency:
    """U_T foi derivado por integração; -grad U deve devolver a Eq. 5.54."""

    @pytest.mark.parametrize("theta", [0, 37, 90, 143, 250])
    def test_negative_gradient_is_the_expanded_force(self, theta):
        p = surface_point(theta)
        step = 1.0  # metro; U varia suavemente nessa escala
        grad = np.empty(3)
        for i in range(3):
            hi, lo = p.copy(), p.copy()
            hi[i] += step
            lo[i] -= step
            grad[i] = (
                tidal_potential(hi, MOON, GM_MOON) - tidal_potential(lo, MOON, GM_MOON)
            ) / (2 * step)

        analytic = tidal_accel_expanded(p, MOON, GM_MOON)
        np.testing.assert_allclose(-grad, analytic, rtol=1e-6, atol=1e-18)


class TestBroadcasting:
    def test_grid_of_points(self):
        pts = np.stack([surface_point(t) for t in range(0, 360, 10)])
        a = tidal_accel_expanded(pts, MOON, GM_MOON)
        assert a.shape == (36, 3)
        # confere um elemento contra o caso escalar
        np.testing.assert_allclose(
            a[9], tidal_accel_expanded(surface_point(90), MOON, GM_MOON)
        )

    def test_trace_is_zero(self):
        """Somar as 3 componentes diagonais do tensor de maré dá zero.

        A força de maré é o gradiente de um potencial harmônico
        (2x, -y, -z -> 2 - 1 - 1 = 0), reflexo de que nao há massa local.
        """
        k = GM_MOON / A_MOON**3
        diag = []
        for i in range(3):
            e = np.zeros(3)
            e[i] = R_EARTH
            diag.append(tidal_accel_expanded(e, MOON, GM_MOON)[i] / (k * R_EARTH))
        assert sum(diag) == pytest.approx(0.0, abs=1e-12)
