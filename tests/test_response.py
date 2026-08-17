"""Resposta da bacia: regime permanente analítico vs. RK4."""

from __future__ import annotations

import numpy as np
import pytest

from oceantides.constants import T_M2
from oceantides.response import (
    BasinResponse,
    quality_from_amplification,
    steady_state_response,
)


class TestSteadyStateFormula:
    def test_slow_forcing_follows_equilibrium(self):
        """omega << omega0: a bacia acompanha o equilíbrio, A -> 1, atraso -> 0."""
        amp, lag = steady_state_response(1.0, 5.0, 1e-4)
        assert float(amp) == pytest.approx(1.0, abs=1e-6)
        assert float(lag) == pytest.approx(0.0, abs=1e-3)

    def test_fast_forcing_is_suppressed_and_antiphase(self):
        """omega >> omega0: a bacia não consegue responder, A -> 0, atraso -> pi."""
        amp, lag = steady_state_response(1.0, 5.0, 100.0)
        assert float(amp) < 1e-3
        assert float(lag) == pytest.approx(np.pi, abs=1e-2)

    def test_at_resonance_amplification_is_q_and_lag_is_quarter_cycle(self):
        amp, lag = steady_state_response(1.0, 12.0, 1.0)
        assert float(amp) == pytest.approx(12.0, rel=1e-12)
        assert float(lag) == pytest.approx(np.pi / 2, rel=1e-12)

    def test_lag_is_always_positive(self):
        """A resposta nunca antecede o forçamento -- causalidade."""
        omega = np.geomspace(0.01, 100.0, 200)
        _, lag = steady_state_response(1.0, 5.0, omega)
        assert np.all(lag > 0) and np.all(lag < np.pi)

    def test_quality_roundtrip(self):
        omega0, omega = 2 * np.pi / (12.5 * 3600), 2 * np.pi / T_M2
        q = quality_from_amplification(omega0, omega, 8.0)
        amp, _ = steady_state_response(omega0, q, omega)
        assert float(amp) == pytest.approx(8.0, rel=1e-9)


class TestRK4MatchesSteadyState:
    """O RK4 integrado deve convergir para a solução analítica de regime.

    É a validação cruzada mais forte do módulo: o integrador e a fórmula
    fechada são caminhos independentes para a mesma resposta.
    """

    @pytest.mark.parametrize(
        "period_h,q", [(12.5, 20.0), (13.5, 8.0), (7.0, 2.0), (20.0, 1.0)]
    )
    def test_amplitude_and_lag(self, period_h, q):
        basin = BasinResponse([period_h * 3600.0], [q])
        omega = 2 * np.pi / T_M2

        forcing = lambda t: np.array([np.cos(omega * t)])

        dt = 60.0
        # 12 tau em vez dos 6 padrão: aqui comparamos com a fórmula EXATA de
        # regime, e 6 tau deixariam exp(-6) ~ 0.25% de transiente residual --
        # o suficiente para mascarar o erro do próprio integrador
        n_spin = int(basin.spin_up_duration(n_tau=12.0) / dt)
        _, h_s, v_s = basin.integrate(forcing, 0.0, dt, n_spin, sample_every=n_spin)

        n = int(round(4 * T_M2 / dt))
        t, h, _ = basin.integrate(
            forcing, n_spin * dt, dt, n, state0=np.stack([h_s[-1], v_s[-1]])
        )
        h = h[:, 0]

        amp_exact, lag_exact = steady_state_response(basin.omega0, q, omega)

        assert h.max() == pytest.approx(amp_exact[0], rel=2e-3)

        # h(t) = A cos(wt - delta)  ->  c = A cos(delta), s = A sin(delta)
        c = 2 * np.mean(h * np.cos(omega * t))
        s = 2 * np.mean(h * np.sin(omega * t))
        assert np.hypot(c, s) == pytest.approx(amp_exact[0], rel=2e-3)
        assert np.arctan2(s, c) == pytest.approx(lag_exact[0], abs=3e-3)


class TestVectorisation:
    def test_many_basins_advance_independently(self):
        periods = np.linspace(6.0, 20.0, 50) * 3600.0
        qs = np.linspace(1.0, 20.0, 50)
        basin = BasinResponse(periods, qs)

        omega = 2 * np.pi / T_M2
        forcing = lambda t: np.full(50, np.cos(omega * t))

        dt = 120.0
        n_spin = int(basin.spin_up_duration() / dt)
        _, h_s, v_s = basin.integrate(forcing, 0.0, dt, n_spin, sample_every=n_spin)
        # retoma em n_spin*dt, e não em 0: o forçamento é cos(wt) em tempo
        # absoluto, então reiniciar o relógio reintroduziria um transiente
        _, h, _ = basin.integrate(
            forcing, n_spin * dt, dt, int(4 * T_M2 / dt),
            state0=np.stack([h_s[-1], v_s[-1]]),
        )

        expected, _ = steady_state_response(basin.omega0, qs, omega)
        np.testing.assert_allclose(h.max(axis=0), expected, rtol=5e-3)

    def test_decay_time_is_q_over_pi_periods(self):
        """tau = 2Q/omega0, ou seja Q/pi periodos naturais -- nao Q."""
        basin = BasinResponse([12.0 * 3600.0], [10.0])
        assert basin.decay_time[0] == pytest.approx(
            10.0 / np.pi * 12.0 * 3600.0, rel=1e-12
        )


class TestForcingMustBeCallable:
    """Se h_eq fosse pré-amostrada só em t, o RK4 cairia para 2ª ordem."""

    def test_half_step_evaluation_preserves_fourth_order(self):
        basin = BasinResponse([12.0 * 3600.0], [5.0])
        omega = 2 * np.pi / T_M2
        forcing = lambda t: np.array([np.cos(omega * t)])

        errs = []
        for dt in (900.0, 450.0, 225.0):
            n = int(round(2 * T_M2 / dt))
            _, h, _ = basin.integrate(forcing, 0.0, dt, n)
            _, ref, _ = basin.integrate(forcing, 0.0, dt / 16, n * 16,
                                        sample_every=16)
            errs.append(abs(h[-1, 0] - ref[-1, 0]))

        slope = np.polyfit(np.log([900.0, 450.0, 225.0]), np.log(errs), 1)[0]
        assert slope > 3.5, f"ordem observada {slope:.2f}, esperada ~4"
