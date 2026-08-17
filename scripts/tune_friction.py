"""Calibra a dissipação do modelo contra marégrafos reais.

O modelo barotrópico não resolve a conversão para maré interna sobre topografia
rugosa, que Egbert & Ray (2000) estimam em ~1 TW, ou 25-30% da dissipação total
de maré. A prática corrente é representá-la por um arrasto linear. O valor certo
desse coeficiente não é derivável do modelo -- tem que ser **calibrado**.

Este script varre o coeficiente e mede o RMS contra as constantes harmônicas de
M2 de estações insulares da NOAA, que amostram a maré de oceano aberto.

Uso: ``uv run python scripts/tune_friction.py``
"""

from __future__ import annotations

import time

import numpy as np

from oceantides.lte import LaplaceTidalModel, LTEConfig
from oceantides.validation import OCEAN_STATIONS, compare, load_reference

CANDIDATES = (0.0, 3e-6, 1e-5, 2e-5, 3e-5, 5e-5)
RESOLUTION = 1.0
SPIN_UP = 10.0
ANALYSIS = 2.5


def main():
    load_reference(verbose=False)  # garante o cache antes de medir tempo
    print(f"Calibração do arrasto de maré interna a {RESOLUTION}°")
    print(f"forçante monocromática M2 · spin-up {SPIN_UP:g} d · análise {ANALYSIS:g} d\n")
    print(f"{'r [1/s]':>10}{'RMS ilhas [cm]':>17}{'vies [cm]':>12}"
          f"{'max global [m]':>16}{'tempo [s]':>11}")
    print("-" * 66)

    best = None
    for r in CANDIDATES:
        cfg = LTEConfig(
            resolution=RESOLUTION,
            days=SPIN_UP + ANALYSIS,
            spin_up_days=SPIN_UP,
            forcing="constituent",
            constituent="M2",
            internal_drag=r,
            sample_every=8,
        )
        t0 = time.perf_counter()
        solution = LaplaceTidalModel(cfg).run(constituents=["M2"])
        elapsed = time.perf_counter() - t0

        rows = compare(solution, "M2", OCEAN_STATIONS)
        errs = np.array([e for _, _, m, e in rows if np.isfinite(m)])
        amp, _ = solution.constituent("M2")

        rms = float(np.sqrt(np.mean(errs**2))) * 100
        bias = float(errs.mean()) * 100
        print(f"{r:>10.1e}{rms:>17.2f}{bias:>12.2f}"
              f"{float(np.nanmax(amp)):>16.2f}{elapsed:>11.0f}")

        if best is None or rms < best[1]:
            best = (r, rms)

    print(f"\nmelhor: r = {best[0]:.1e} 1/s  com RMS {best[1]:.2f} cm")
    print("(referência: modelos globais assimilados atingem ~0.9 cm em oceano profundo)")


if __name__ == "__main__":
    main()
