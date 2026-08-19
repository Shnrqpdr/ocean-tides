"""Integradores de EDO.

Duas famílias, porque o projeto tem dois subsistemas com exigências distintas:

**Genérico** ``y' = f(t, y)`` -- :func:`rk4_step`. É o que a resposta dinâmica
do oceano precisa: o amortecimento ``2 gamma h'`` depende da velocidade. O
sistema deixa de ser hamiltoniano, então não há estrutura simplética a
preservar -- os métodos simpléticos perdem a vantagem, e deixam de ser
explícitos porque a força passa a depender da velocidade que estão avançando.

**Separável** ``x' = v``, ``v' = a(t, x)`` -- :func:`integrate_velocity_verlet`
e :func:`integrate_yoshida4`, ambos simpléticos e reversíveis no tempo. É o que
as órbitas precisam: o erro de energia fica *limitado* em vez de crescer
secularmente, como acontece com o RK4.

Cada driver expõe ``.order`` (ordem de convergência) e ``.evals_per_step``
(avaliações de força por passo) para que a comparação em :mod:`oceantides.bench`
seja feita a custo computacional equivalente.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "rk4_step",
    "integrate_ode",
    "integrate_rk4",
    "integrate_velocity_verlet",
    "integrate_yoshida4",
    "reference_solution",
    "SYMPLECTIC_DRIVERS",
    "ORBIT_DRIVERS",
]


# ---------------------------------------------------------------------------
# Genérico: y' = f(t, y)
# ---------------------------------------------------------------------------


def rk4_step(f: Callable, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Um passo de Runge-Kutta clássico de 4ª ordem.

    ``f`` pode devolver qualquer forma que faça broadcast com ``y``, o que
    permite integrar milhares de osciladores independentes de uma só vez.

    Cuidado com ``f``: as avaliações em ``t + dt/2`` são o que dá a 4ª ordem.
    Se o forçamento externo for amostrado só nos instantes ``t``, a ordem cai
    para 2 -- por isso :mod:`oceantides.response` recebe ``h_eq`` como função
    do tempo, e não como um vetor pré-amostrado.
    """
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_ode(f, t0, y0, dt, n_steps, sample_every=1):
    """Integra ``y' = f(t, y)`` com RK4 e devolve ``(t, y)`` amostrados."""
    y = np.array(y0, dtype=float)
    t = float(t0)

    n_out = n_steps // sample_every + 1
    ts = np.empty(n_out)
    ys = np.empty((n_out,) + y.shape)
    ts[0], ys[0] = t, y

    k = 1
    for i in range(1, n_steps + 1):
        y = rk4_step(f, t, y, dt)
        t = t0 + i * dt  # recomputa em vez de acumular: evita deriva de arredondamento
        if i % sample_every == 0:
            ts[k], ys[k] = t, y
            k += 1
    return ts[:k], ys[:k]


# ---------------------------------------------------------------------------
# Separável: x' = v, v' = a(t, x)
# ---------------------------------------------------------------------------


def _prepare(x0, v0, n_steps, sample_every):
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    n_out = n_steps // sample_every + 1
    return x, v, np.empty(n_out), np.empty((n_out,) + x.shape), np.empty((n_out,) + x.shape)


def integrate_rk4(accel, x0, v0, t0, dt, n_steps, sample_every=1):
    """RK4 aplicado ao sistema de 1ª ordem equivalente.

    Incluído para comparação: **não** é simplético, então a energia deriva de
    forma secular. Em órbitas longas isso encolhe o semieixo maior, e como a
    maré vai com ``D^-3`` o erro relativo é amplificado 3x.
    """
    x, v, ts, xs, vs = _prepare(x0, v0, n_steps, sample_every)
    ts[0], xs[0], vs[0] = t0, x, v

    k = 1
    for i in range(1, n_steps + 1):
        t = t0 + (i - 1) * dt

        k1x, k1v = v, accel(t, x)
        k2x, k2v = v + 0.5 * dt * k1v, accel(t + 0.5 * dt, x + 0.5 * dt * k1x)
        k3x, k3v = v + 0.5 * dt * k2v, accel(t + 0.5 * dt, x + 0.5 * dt * k2x)
        k4x, k4v = v + dt * k3v, accel(t + dt, x + dt * k3x)

        x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        if i % sample_every == 0:
            ts[k], xs[k], vs[k] = t0 + i * dt, x, v
            k += 1
    return ts[:k], xs[:k], vs[:k]


integrate_rk4.order = 4
integrate_rk4.evals_per_step = 4
integrate_rk4.symplectic = False
integrate_rk4.label = "RK4"


def integrate_velocity_verlet(accel, x0, v0, t0, dt, n_steps, sample_every=1):
    """Velocity-Verlet: 2ª ordem, simplético, reversível no tempo.

    Uma única avaliação de força por passo (a aceleração do fim de um passo é
    reaproveitada no início do seguinte) -- quatro vezes mais barato que o RK4.
    Exige que ``a`` **não** dependa da velocidade.
    """
    x, v, ts, xs, vs = _prepare(x0, v0, n_steps, sample_every)
    ts[0], xs[0], vs[0] = t0, x, v

    a = accel(t0, x)
    k = 1
    for i in range(1, n_steps + 1):
        t = t0 + (i - 1) * dt
        x = x + v * dt + 0.5 * a * dt * dt
        a_new = accel(t + dt, x)
        v = v + 0.5 * (a + a_new) * dt
        a = a_new

        if i % sample_every == 0:
            ts[k], xs[k], vs[k] = t0 + i * dt, x, v
            k += 1
    return ts[:k], xs[:k], vs[:k]


integrate_velocity_verlet.order = 2
integrate_velocity_verlet.evals_per_step = 1
integrate_velocity_verlet.symplectic = True
integrate_velocity_verlet.label = "Velocity-Verlet"


# Coeficientes de Yoshida (1990): composição tripla de saltos de Verlet que
# cancela o termo de erro de 3ª ordem.
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 / (2.0 - _CBRT2)
_YOSHIDA_C = (_W1 / 2.0, (_W0 + _W1) / 2.0, (_W0 + _W1) / 2.0, _W1 / 2.0)
_YOSHIDA_D = (_W1, _W0, _W1)


def integrate_yoshida4(accel, x0, v0, t0, dt, n_steps, sample_every=1):
    """Yoshida de 4ª ordem: simplético, mesma ordem do RK4, sem deriva secular.

    Três avaliações de força por passo (contra quatro do RK4), e o erro de
    energia oscila em torno de zero em vez de crescer. É a opção certa quando
    se quer 4ª ordem *e* estabilidade em integrações longas.
    """
    x, v, ts, xs, vs = _prepare(x0, v0, n_steps, sample_every)
    ts[0], xs[0], vs[0] = t0, x, v

    k = 1
    for i in range(1, n_steps + 1):
        t = t0 + (i - 1) * dt
        elapsed = 0.0
        for j in range(4):
            x = x + _YOSHIDA_C[j] * v * dt
            elapsed += _YOSHIDA_C[j] * dt
            if j < 3:
                v = v + _YOSHIDA_D[j] * accel(t + elapsed, x) * dt

        if i % sample_every == 0:
            ts[k], xs[k], vs[k] = t0 + i * dt, x, v
            k += 1
    return ts[:k], xs[:k], vs[:k]


integrate_yoshida4.order = 4
integrate_yoshida4.evals_per_step = 3
integrate_yoshida4.symplectic = True
integrate_yoshida4.label = "Yoshida-4"


SYMPLECTIC_DRIVERS = {
    "verlet": integrate_velocity_verlet,
    "yoshida4": integrate_yoshida4,
}

ORBIT_DRIVERS = {
    "rk4": integrate_rk4,
    **SYMPLECTIC_DRIVERS,
}


# ---------------------------------------------------------------------------
# Referência de alta precisão
# ---------------------------------------------------------------------------


def reference_solution(accel, x0, v0, t_eval, rtol=1e-12, atol=1e-12):
    """Solução de referência via Dormand-Prince 8(5,3) do SciPy.

    Usada nos testes como verdade independente quando não há solução analítica.
    """
    from scipy.integrate import solve_ivp

    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    n = x0.size

    def rhs(t, y):
        return np.concatenate([y[n:], np.ravel(accel(t, y[:n].reshape(x0.shape)))])

    t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(
        rhs,
        (t_eval[0], t_eval[-1]),
        np.concatenate([np.ravel(x0), np.ravel(v0)]),
        t_eval=t_eval,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp falhou: {sol.message}")
    return sol.y[:n].T.reshape((-1,) + x0.shape), sol.y[n:].T.reshape((-1,) + x0.shape)
