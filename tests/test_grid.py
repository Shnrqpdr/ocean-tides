"""Grade geográfica, rotação da Terra e o campo global."""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.constants import (
    GM_MOON,
    OMEGA_EARTH,
    R_EARTH,
    T_SIDEREAL_DAY,
)
from oceantides.ephemeris import moon_ephemeris
from oceantides.forcing import equilibrium_height
from oceantides.grid import (
    Grid,
    eci_to_ecef,
    geographic_to_ecef,
    greenwich_angle,
    subbody_point,
)


class TestFrames:
    def test_rotation_preserves_length_and_z(self):
        v = np.array([1.0, 2.0, 3.0])
        out = eci_to_ecef(v, 12345.0)
        assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(v))
        assert out[2] == pytest.approx(v[2])

    def test_identity_at_t_zero(self):
        v = np.array([1.0, -2.0, 0.5])
        np.testing.assert_allclose(eci_to_ecef(v, 0.0), v, atol=1e-12)

    def test_full_sidereal_day_returns_to_start(self):
        v = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(eci_to_ecef(v, T_SIDEREAL_DAY), v, atol=1e-9)

    def test_half_sidereal_day_flips_equatorial_components(self):
        v = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            eci_to_ecef(v, T_SIDEREAL_DAY / 2), [-1.0, 0.0, 0.0], atol=1e-9
        )

    def test_greenwich_angle_rate(self):
        assert greenwich_angle(1.0) - greenwich_angle(0.0) == pytest.approx(OMEGA_EARTH)


class TestGeographic:
    @pytest.mark.parametrize(
        "lat,lon,expected",
        [
            (0, 0, [R_EARTH, 0, 0]),
            (0, 90, [0, R_EARTH, 0]),
            (90, 0, [0, 0, R_EARTH]),
            (-90, 0, [0, 0, -R_EARTH]),
            (0, 180, [-R_EARTH, 0, 0]),
        ],
    )
    def test_known_points(self, lat, lon, expected):
        np.testing.assert_allclose(geographic_to_ecef(lat, lon), expected, atol=1e-6)

    def test_all_points_on_the_sphere(self):
        g = Grid(31, 61)
        radii = np.linalg.norm(g.points, axis=-1)
        np.testing.assert_allclose(radii, R_EARTH, rtol=1e-12)


class TestSubBodyPoint:
    def test_body_on_x_axis_at_t_zero(self):
        lat, lon = subbody_point(np.array([1e8, 0.0, 0.0]), 0.0)
        assert float(lat) == pytest.approx(0.0, abs=1e-9)
        assert float(lon) == pytest.approx(0.0, abs=1e-9)

    def test_longitude_regresses_as_earth_turns(self):
        """O ponto sublunar caminha para oeste conforme a Terra gira."""
        body = np.array([1e8, 0.0, 0.0])
        _, lon0 = subbody_point(body, 0.0)
        _, lon1 = subbody_point(body, 3600.0)
        assert float(lon1) < float(lon0)
        assert float(lon0) - float(lon1) == pytest.approx(
            np.degrees(OMEGA_EARTH * 3600.0), abs=1e-6
        )

    def test_latitude_equals_declination(self):
        body = np.array([0.0, 0.0, 1e8])
        lat, _ = subbody_point(body, 12345.0)
        assert float(lat) == pytest.approx(90.0)

    def test_moon_declination_stays_within_28_6_degrees(self):
        """Inclinação orbital 5.145 graus + obliquidade 23.44 -> +-28.6."""
        moon = moon_ephemeris("kepler")
        t = np.linspace(0.0, 30 * 86400.0, 4000)
        lat, _ = subbody_point(moon.position(t), t)
        assert np.abs(lat).max() == pytest.approx(28.6, abs=0.2)


@pytest.fixture(scope="module")
def setup():
    return Grid(46, 91), [(moon_ephemeris("kepler"), GM_MOON)]


class TestEquilibriumField:

    def test_matches_direct_computation(self, setup):
        grid, bodies = setup
        t = 5 * 86400.0
        field = grid.equilibrium_field(bodies, t)
        direct = equilibrium_height(
            grid.points, eci_to_ecef(bodies[0][0].position(t), t), GM_MOON
        )
        np.testing.assert_allclose(field, direct, rtol=1e-12)

    def test_area_mean_is_zero(self, setup):
        """A maré redistribui água, não a cria (p. 202: dois bojos, dois vales)."""
        grid, bodies = setup
        for t in (0.0, 3.7e5, 1.1e6):
            field = grid.equilibrium_field(bodies, t)
            assert abs(grid.area_mean(field)) < 1e-3 * np.ptp(field)

    def test_two_maxima_around_the_equator(self, setup):
        """Dois bojos antipodais -> a Terra girando encontra duas marés altas."""
        grid, bodies = setup
        field = grid.equilibrium_field(bodies, 0.0)
        equator = field[grid.n_lat // 2, :-1]  # exclui a longitude duplicada
        interior = equator[1:-1]
        n_peaks = int(((interior > equator[:-2]) & (interior > equator[2:])).sum())
        wraps = equator[0] > equator[1] and equator[0] > equator[-1]
        assert n_peaks + int(wraps) == 2

    def test_superposition_is_linear(self, setup):
        """A maré é gradiente de um potencial, então Lua e Sol simplesmente somam.

        É isso que produz sizígia e quadratura sem nenhum tratamento especial.
        """
        from oceantides.constants import GM_SUN
        from oceantides.ephemeris import sun_ephemeris

        grid, moon_only = setup
        sun_only = [(sun_ephemeris("kepler"), GM_SUN)]
        t = 2.5e5
        both = grid.equilibrium_field(moon_only + sun_only, t)
        np.testing.assert_allclose(
            both,
            grid.equilibrium_field(moon_only, t) + grid.equilibrium_field(sun_only, t),
            rtol=1e-12,
        )

    def test_peak_sits_at_the_subbody_point(self, setup):
        grid, bodies = setup
        t = 4.2e5
        field = grid.equilibrium_field(bodies, t)
        j, k = np.unravel_index(np.argmax(field), field.shape)
        lat, lon = subbody_point(bodies[0][0].position(t), t)

        # o máximo fica no ponto sublunar OU no seu antípoda
        d_direct = abs(grid.lat[j] - float(lat))
        d_anti = abs(grid.lat[j] + float(lat))
        assert min(d_direct, d_anti) < 5.0
