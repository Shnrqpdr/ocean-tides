"""Ordem de convergência, simpleticidade e a equação de Kepler."""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.constants import T_MOON_SIDEREAL
from oceantides.ephemeris import NBodyEphemeris, moon_ephemeris, solve_kepler
from oceantides.integrators import (
    ORBIT_DRIVERS,
    integrate_ode,
    reference_solution,
    rk4_step,
)

YEAR = 365.25 * 86400.0


@pytest.fixture(scope="module")
def moon():
    return moon_ephemeris("kepler")


class TestKeplerEquation:
    @pytest.mark.parametrize("e", [0.0, 0.0167, 0.0549, 0.3, 0.7])
    def test_residual_is_machine_zero(self, e):
        M = np.linspace(-np.pi, np.pi, 501)
        E = solve_kepler(M, e)
        residual = E - e * np.sin(E) - np.mod(M + np.pi, 2 * np.pi) + np.pi
        assert np.max(np.abs(residual)) < 1e-12

    def test_circular_case_is_identity(self):
        M = np.linspace(0, 2 * np.pi, 17)
        np.testing.assert_allclose(solve_kepler(M, 0.0), np.mod(M + np.pi, 2 * np.pi) - np.pi)


class TestConvergenceOrder:
    """Cada driver deve exibir a ordem que anuncia em ``.order``."""

    @pytest.mark.parametrize("name", list(ORBIT_DRIVERS))
    def test_observed_order_matches_declared(self, name, moon):
        driver = ORBIT_DRIVERS[name]
        duration = T_MOON_SIDEREAL / 4

        dts, errs = [], []
        for dt in (14400.0, 7200.0, 3600.0, 1800.0):
            nb = NBodyEphemeris(moon, method=name, dt=dt, duration=duration)
            errs.append(np.linalg.norm(nb.x[-1] - moon.position(nb.t[-1])))
            dts.append(dt)

        slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        assert slope == pytest.approx(driver.order, abs=0.1)


class TestSymplecticity:
    """A distinção que importa: erro de energia *limitado* vs *secular*.

    Medido em 10 anos de órbita lunar a dt = 1h. O RK4 acumula erro que cresce
    proporcionalmente ao tempo; Verlet e Yoshida-4 oscilam dentro de um teto.
    """

    @staticmethod
    def _energy_drift(name, moon, years=10):
        nb = NBodyEphemeris(moon, method=name, dt=3600.0, duration=years * YEAR)
        e = nb.specific_energy()
        return nb.t, np.abs((e - e[0]) / e[0])

    @pytest.mark.parametrize("name", ["verlet", "yoshida4"])
    def test_symplectic_error_is_bounded(self, name, moon):
        t, drift = self._energy_drift(name, moon)
        one_year = drift[np.argmin(np.abs(t - YEAR))]
        ten_years = drift[np.argmin(np.abs(t - 10 * YEAR))]
        # 10x o tempo não deve significar 10x o erro
        assert ten_years / one_year < 2.0

    def test_rk4_error_is_secular(self, moon):
        t, drift = self._energy_drift("rk4", moon)
        one_year = drift[np.argmin(np.abs(t - YEAR))]
        ten_years = drift[np.argmin(np.abs(t - 10 * YEAR))]
        # cresce quase linearmente com o tempo
        assert ten_years / one_year > 5.0

    def test_rk4_is_still_more_accurate_than_verlet(self, moon):
        """Nuance importante: simplético não implica mais preciso.

        Verlet é simplético mas de 2ª ordem, e ao longo de 10 anos continua
        ~1000x pior que o RK4 de 4ª ordem. A vantagem simplética é o *formato*
        do erro, não a magnitude.
        """
        _, rk4 = self._energy_drift("rk4", moon)
        _, verlet = self._energy_drift("verlet", moon)
        assert rk4.max() < verlet.max() / 100

    def test_yoshida4_gets_both(self, moon):
        """Yoshida-4 é 4ª ordem *e* limitado -- domina o RK4 em 10 anos."""
        _, rk4 = self._energy_drift("rk4", moon)
        _, yoshida = self._energy_drift("yoshida4", moon)
        assert yoshida[-1] < rk4[-1]


class TestTidalImpact:
    """Traduz erro numérico em erro físico: dh/h = 3 dD/D, pois h ~ D^-3."""

    @pytest.mark.parametrize("name", list(ORBIT_DRIVERS))
    def test_tide_error_is_negligible_over_a_year(self, name, moon):
        nb = NBodyEphemeris(moon, method=name, dt=3600.0, duration=YEAR)
        d_num = np.linalg.norm(nb.x, axis=-1)
        d_exact = moon.distance(nb.t)
        tide_err = 3.0 * np.max(np.abs(d_num - d_exact) / d_exact)
        # mesmo o pior caso fica muito abaixo de 0.1% da amplitude da maré
        assert tide_err < 1e-3


class TestGenericRK4:
    """rk4_step sobre y' = f(t, y), que é a forma usada pela resposta dinâmica."""

    def test_exponential_decay(self):
        f = lambda t, y: -0.5 * y
        _, ys = integrate_ode(f, 0.0, np.array([1.0]), 0.01, 1000)
        assert ys[-1, 0] == pytest.approx(np.exp(-5.0), rel=1e-9)

    def test_harmonic_oscillator_amplitude(self):
        """Sem amortecimento a amplitude deve se conservar (a menos do erro RK4)."""
        w = 2 * np.pi
        f = lambda t, y: np.array([y[1], -(w**2) * y[0]])
        _, ys = integrate_ode(f, 0.0, np.array([1.0, 0.0]), 1e-3, 10_000)
        amp = np.hypot(ys[:, 0], ys[:, 1] / w)
        assert np.max(np.abs(amp - 1.0)) < 1e-8

    def test_fourth_order_on_nonautonomous_forcing(self):
        """Confirma que f é avaliada em t + dt/2 (senão a ordem cai para 2)."""
        f = lambda t, y: np.array([np.cos(t)])  # solução exata: sin(t)
        errs = []
        for dt in (0.2, 0.1, 0.05):
            _, ys = integrate_ode(f, 0.0, np.array([0.0]), dt, int(round(10.0 / dt)))
            errs.append(abs(ys[-1, 0] - np.sin(10.0)))
        slope = np.polyfit(np.log([0.2, 0.1, 0.05]), np.log(errs), 1)[0]
        assert slope == pytest.approx(4.0, abs=0.2)

    def test_broadcasts_over_many_oscillators(self):
        """Um único rk4_step deve avançar 1000 osciladores independentes."""
        w = np.linspace(1.0, 2.0, 1000)
        y = np.zeros((2, 1000))
        y[0] = 1.0
        f = lambda t, s: np.stack([s[1], -(w**2) * s[0]])
        for i in range(100):
            y = rk4_step(f, i * 1e-3, y, 1e-3)
        expected = np.cos(w * 0.1)
        np.testing.assert_allclose(y[0], expected, rtol=1e-9)


class TestBenchReporting:
    """O relatório do `bench` precisa medir o que diz medir.

    Regressão: comparar contra `exact.position(duration)` em vez do instante
    final efetivo `n_steps*dt` injeta um erro de até meio passo -- milhões de
    metros à velocidade orbital -- que mascara o erro do integrador e faz a
    ordem observada colapsar para ~0.
    """

    def test_reported_orders_match_the_declared_ones(self):
        from oceantides.bench import convergence_study

        for label, info in convergence_study().items():
            assert info["observed_order"] == pytest.approx(
                info["declared_order"], abs=0.1
            ), f"{label}: observada {info['observed_order']:.3f}"

    def test_errors_decrease_with_step_size(self):
        from oceantides.bench import convergence_study

        for label, info in convergence_study().items():
            errs = info["errors"]
            assert all(a > b for a, b in zip(errs, errs[1:])), label


class TestReferenceSolution:
    def test_dop853_matches_kepler(self, moon):
        accel = lambda t, x: -moon.mu * x / np.linalg.norm(x) ** 3
        t_eval = np.linspace(0.0, T_MOON_SIDEREAL, 50)
        xs, _ = reference_solution(accel, moon.position(0.0), moon.velocity(0.0), t_eval)
        err = np.max(np.linalg.norm(xs - moon.position(t_eval), axis=-1))
        assert err < 1.0  # metro, numa órbita de 384 mil km
