"""Resposta dinâmica de uma bacia oceânica ao forçamento de maré.

O PDF encerra a seção (p. 203) observando que as marés costeiras reais são bem
maiores que os 0.54 m da teoria de equilíbrio, e que *"resonances can affect the
natural oscillation of the bodies of water and cause tidal changes"*. Este
módulo formaliza essa observação com o modelo mínimo que a captura:

.. math::

    \\ddot{h} + 2\\gamma\\dot{h} + \\omega_0^2 h = \\omega_0^2 h_{eq}(t)

Uma bacia com período natural ``T_0 = 2 pi / omega_0`` e fator de qualidade
``Q = omega_0 / (2 gamma)``, forçada pela maré de equilíbrio. Em regime
permanente a resposta a uma componente de frequência ``omega`` é amplificada
por ``A(omega)`` e **atrasada** por ``delta(omega)`` -- e é esse atraso que
produz a Figura 5-13, em que a maré alta não está sobre o eixo Terra-Lua.

Este é o subsistema onde o RK4 é de fato a escolha certa: o termo ``2 gamma
h'`` depende da velocidade, o que quebra a estrutura separável de que os
integradores simpléticos dependem.
"""

from __future__ import annotations

import numpy as np

from .integrators import rk4_step

__all__ = [
    "steady_state_response",
    "BasinResponse",
    "quality_from_amplification",
]


def steady_state_response(omega0, q_factor, omega):
    """Amplificação e atraso de fase em regime permanente.

    Returns
    -------
    amplification : ndarray
        ``A = omega0^2 / sqrt((omega0^2 - omega^2)^2 + 4 gamma^2 omega^2)``.
        Vale 1 no limite de forçamento lento e cresce até ``~Q`` na ressonância.
    lag : ndarray
        Atraso de fase [rad], em ``(0, pi)``. Multiplicado por ``T/2pi`` dá o
        atraso em horas com que a maré alta chega depois da passagem da Lua.
    """
    omega0 = np.asarray(omega0, dtype=float)
    omega = np.asarray(omega, dtype=float)
    gamma = omega0 / (2.0 * np.asarray(q_factor, dtype=float))

    denom = np.sqrt((omega0**2 - omega**2) ** 2 + 4.0 * gamma**2 * omega**2)
    amplification = omega0**2 / denom
    lag = np.arctan2(2.0 * gamma * omega, omega0**2 - omega**2)
    return amplification, lag


def quality_from_amplification(omega0, omega, amplification):
    """Q que produz uma dada amplificação -- útil para calibrar uma bacia."""
    omega0, omega = float(omega0), float(omega)
    target = (omega0**2 / amplification) ** 2 - (omega0**2 - omega**2) ** 2
    if target <= 0:
        raise ValueError("amplificação inatingível para esses omega0/omega")
    gamma = np.sqrt(target) / (2.0 * omega)
    return omega0 / (2.0 * gamma)


class BasinResponse:
    """Conjunto de osciladores independentes, integrados em bloco com RK4.

    Todas as bacias avançam simultaneamente como um único array NumPy de forma
    ``(2, n)``, então o custo por passo é o de umas poucas operações vetoriais,
    independentemente de haver 5 ou 5000 estações.
    """

    def __init__(self, natural_period, q_factor):
        self.omega0 = 2.0 * np.pi / np.atleast_1d(np.asarray(natural_period, float))
        self.q_factor = np.broadcast_to(
            np.atleast_1d(np.asarray(q_factor, float)), self.omega0.shape
        ).copy()
        self.gamma = self.omega0 / (2.0 * self.q_factor)

    @property
    def n(self) -> int:
        return self.omega0.size

    def derivative(self, t, state, forcing):
        """``y' = f(t, y)`` com ``y = [h, dh/dt]``.

        ``forcing`` é uma **função de t**, não um array pré-amostrado: o RK4
        precisa avaliá-la em ``t + dt/2``, e amostrá-la apenas nos instantes
        ``t`` derrubaria a ordem do método de 4 para 2.
        """
        h, h_dot = state[0], state[1]
        h_eq = forcing(t)
        return np.stack(
            [h_dot, self.omega0**2 * (h_eq - h) - 2.0 * self.gamma * h_dot]
        )

    def integrate(self, forcing, t0, dt, n_steps, sample_every=1, state0=None):
        """Integra com RK4 e devolve ``(t, h, dh/dt)``.

        A condição inicial padrão é o equilíbrio instantâneo com velocidade
        nula. O transiente decai em ``~Q`` períodos; use
        :meth:`spin_up_steps` para saber quanto descartar.
        """
        state = (
            np.array(state0, dtype=float)
            if state0 is not None
            else np.stack([np.broadcast_to(forcing(t0), (self.n,)).copy(), np.zeros(self.n)])
        )

        n_out = n_steps // sample_every + 1
        ts = np.empty(n_out)
        hs = np.empty((n_out, self.n))
        vs = np.empty((n_out, self.n))
        ts[0], hs[0], vs[0] = t0, state[0], state[1]

        f = lambda t, y: self.derivative(t, y, forcing)

        k = 1
        for i in range(1, n_steps + 1):
            state = rk4_step(f, t0 + (i - 1) * dt, state, dt)
            if i % sample_every == 0:
                ts[k], hs[k], vs[k] = t0 + i * dt, state[0], state[1]
                k += 1
        return ts[:k], hs[:k], vs[:k]

    @property
    def decay_time(self) -> np.ndarray:
        """Constante de tempo ``1/gamma = 2Q/omega0`` do transiente [s].

        Em períodos naturais isso é ``Q/pi``, e não ``Q``: o transiente decai
        como ``exp(-gamma t)`` com ``gamma = omega0/(2Q)``.
        """
        return 2.0 * self.q_factor / self.omega0

    def spin_up_duration(self, n_tau: float = 6.0) -> float:
        """Tempo a integrar *antes* de ``t=0`` para o transiente decair.

        Sem isso a saída começa com o oscilador "subindo" ao regime, o que
        mascara completamente o envelope de sizígia/quadratura nas primeiras
        semanas de uma bacia de Q alto.

        O padrão de 6 tau deixa ``exp(-6) ~ 0.25%`` de transiente residual --
        bem abaixo da incerteza do próprio modelo, e escolhido para não
        encarecer a simulação (uma bacia com Q = 20 tem tau de 3.3 dias, então
        6 tau já custam 20 dias de integração descartada).
        """
        return float(n_tau * np.max(self.decay_time))

    def spin_up_steps(self, dt, n_tau: float = 6.0) -> np.ndarray:
        """Passos equivalentes ao transiente, por oscilador."""
        return np.ceil(n_tau * self.decay_time / dt).astype(int)

    def response_at(self, period_seconds):
        """Atalho para :func:`steady_state_response` num período dado."""
        return steady_state_response(
            self.omega0, self.q_factor, 2.0 * np.pi / period_seconds
        )
