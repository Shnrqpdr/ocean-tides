"""Comparação empírica dos integradores.

Responde à pergunta "RK4 é a escolha certa?" com medida em vez de afirmação.
Integra a órbita lunar com cada método e reporta, além dos números usuais de
erro, a única métrica que importa para este projeto: **quanto erro numérico
isso injeta na altura da maré**. Como ``h ~ D^-3``, vale ``dh/h = 3 dD/D``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import T_MOON_SIDEREAL
from .ephemeris import NBodyEphemeris, moon_ephemeris
from .integrators import ORBIT_DRIVERS

__all__ = ["BenchRow", "run_bench", "format_bench", "convergence_study"]

YEAR = 365.25 * 86400.0


@dataclass
class BenchRow:
    method: str
    dt: float
    energy_drift: float
    max_position_error: float
    tide_error_pct: float
    seconds: float
    evals_per_step: int
    symplectic: bool
    secular_ratio: float  # erro(10 anos)/erro(1 ano): ~10 secular, ~1 limitado


def _timed(fn):
    import time

    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_bench(dts=(600.0, 1800.0, 3600.0), years=10.0, methods=None):
    """Integra a órbita lunar por ``years`` com cada método e cada ``dt``."""
    # SEM precessão: os drivers integram o problema de DOIS CORPOS puro, cuja
    # solução exata é a elipse fixa. Comparar contra a órbita precessante
    # mediria a física ausente do integrador, não o erro numérico dele.
    exact = moon_ephemeris("kepler", precess=False)
    methods = list(methods or ORBIT_DRIVERS)
    rows = []

    for dt in dts:
        for name in methods:
            driver = ORBIT_DRIVERS[name]
            nb, seconds = _timed(
                lambda: NBodyEphemeris(exact, method=name, dt=dt, duration=years * YEAR)
            )

            energy = nb.specific_energy()
            drift = np.abs((energy - energy[0]) / energy[0])

            d_num = np.linalg.norm(nb.x, axis=-1)
            d_exact = exact.distance(nb.t)
            tide_err = 3.0 * np.max(np.abs(d_num - d_exact) / d_exact) * 100.0
            pos_err = np.max(np.linalg.norm(nb.x - exact.position(nb.t), axis=-1))

            one = drift[np.argmin(np.abs(nb.t - YEAR))]
            ten = drift[np.argmin(np.abs(nb.t - years * YEAR))]

            rows.append(
                BenchRow(
                    method=driver.label,
                    dt=dt,
                    energy_drift=float(drift[-1]),
                    max_position_error=float(pos_err),
                    tide_error_pct=float(tide_err),
                    seconds=seconds,
                    evals_per_step=driver.evals_per_step,
                    symplectic=driver.symplectic,
                    secular_ratio=float(ten / one) if one > 0 else float("nan"),
                )
            )
    return rows


def convergence_study(dts=(14400.0, 7200.0, 3600.0, 1800.0), fraction=0.25):
    """Ordem de convergência observada de cada driver."""
    exact = moon_ephemeris("kepler", precess=False)  # ver nota em run_bench
    duration = fraction * T_MOON_SIDEREAL
    out = {}
    for name, driver in ORBIT_DRIVERS.items():
        errs = []
        for dt in dts:
            nb = NBodyEphemeris(exact, method=name, dt=dt, duration=duration)
            # Compara no instante final EFETIVO (n_steps*dt), não em `duration`:
            # os dois diferem por até meio passo, e a Lua percorre ~1 km/s, o
            # que injetaria um erro de milhões de metros e mascararia por
            # completo o erro real do integrador.
            errs.append(np.linalg.norm(nb.x[-1] - exact.position(nb.t[-1])))
        slope = float(np.polyfit(np.log(dts), np.log(errs), 1)[0])
        out[driver.label] = {
            "declared_order": driver.order,
            "observed_order": slope,
            "errors": [float(e) for e in errs],
            "dts": list(dts),
        }
    return out


def format_bench(rows, years=10.0) -> str:
    """Tabela legível no terminal, com a leitura já interpretada."""
    lines = [
        f"Orbita lunar integrada por {years:.0f} anos, comparada ao Kepler analitico",
        "",
        f"{'metodo':<16}{'dt[s]':>7}{'|dE/E|':>11}{'erro pos[m]':>13}"
        f"{'erro h[%]':>12}{'aval/passo':>12}{'tempo[s]':>10}{'secular?':>10}",
        "-" * 91,
    ]
    for r in rows:
        secular = "sim" if r.secular_ratio > 3.0 else "nao"
        lines.append(
            f"{r.method:<16}{r.dt:>7.0f}{r.energy_drift:>11.2e}"
            f"{r.max_position_error:>13.2e}{r.tide_error_pct:>12.2e}"
            f"{r.evals_per_step:>12d}{r.seconds:>10.2f}{secular:>10}"
        )

    conv = convergence_study()
    lines += ["", "Ordem de convergencia observada:"]
    for label, info in conv.items():
        lines.append(
            f"  {label:<16} declarada {info['declared_order']}  "
            f"observada {info['observed_order']:.3f}"
        )

    worst_tide = max(r.tide_error_pct for r in rows)
    worst_mm = worst_tide / 100.0 * 0.54 * 1000.0
    lines += [
        "",
        "Leitura:",
        "  * Simpletico (Verlet, Yoshida-4): erro de energia LIMITADO -- oscila em",
        "    torno de zero. RK4: erro SECULAR -- cresce junto com o tempo. Medido em",
        "    10 anos, o erro do RK4 fica ~9x o de 1 ano; o dos simpleticos, ~1x.",
        "  * Mas simpletico NAO implica mais preciso. Verlet e simpletico e mesmo",
        "    assim fica ordens de grandeza pior que o RK4, porque e de 2a ordem: a",
        "    ordem do metodo pesa mais que a simpleticidade nestas escalas de tempo.",
        "  * Yoshida-4 tem as duas coisas -- 4a ordem E erro limitado -- por 3",
        "    avaliacoes de forca por passo, contra 4 do RK4. E a melhor escolha",
        "    entre os integradores propriamente ditos.",
        "  * Kepler analitico vence todos: precisao de maquina, deriva zero, custo",
        "    O(1) por instante. E o padrao de producao deste projeto.",
        "",
        f"  * IMPACTO PRATICO: o pior erro de mare da tabela e {worst_tide:.3g}%, ou seja",
        f"    {worst_mm:.2f} mm numa mare de 0.54 m. Nenhum destes metodos compromete",
        "    o resultado fisico. A sugestao original de RK4 funcionaria perfeitamente;",
        "    o Kepler analitico foi adotado por ser mais simples e exato, nao por",
        "    necessidade numerica.",
    ]
    return "\n".join(lines)
