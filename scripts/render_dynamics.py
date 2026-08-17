"""Resolve as LTE a 1 grau e gera todos os produtos da parte dinâmica.

Saídas em ``media/``:

* ``carta_cotidal_M2.png`` — a figura clássica da maré, com os anfidromos
* ``validacao_M2.png`` — modelo contra marégrafos da NOAA
* ``mare_onda.mp4`` — LTE contra maré de equilíbrio, lado a lado
* ``mare_correntes.mp4`` — correntes de maré

e a solução harmônica em ``data/lte_m2.npz``, de onde tudo pode ser
reconstruído sem repetir a integração.

Uso: ``uv run --extra data python scripts/render_dynamics.py [--resolution 1.0]``
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from oceantides.cli import _save_animation
from oceantides.io import save_solution
from oceantides.lte import LaplaceTidalModel, LTEConfig
from oceantides.validation import ALL_STATIONS, OCEAN_STATIONS, compare, format_comparison
from oceantides.viz.cotidal import plot_cotidal_chart, plot_validation
from oceantides.viz.tide_wave import animate, animate_currents

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "media"
DATA = ROOT / "data"
DOCS = ROOT / "docs"


def build(resolution, days, spin_up, internal_drag, constituent):
    cfg = LTEConfig(
        resolution=resolution,
        days=days,
        spin_up_days=spin_up,
        forcing="constituent",
        constituent=constituent,
        internal_drag=internal_drag,
        sample_every=8,
    )
    model = LaplaceTidalModel(cfg)
    print(model.summary(), flush=True)

    def progress(k, total, _s):
        print(f"  integrando {100 * k / total:5.1f}%", end="\r", flush=True)

    t0 = time.perf_counter()
    solution = model.run(constituents=[constituent], progress=progress)
    print(f"  integrado em {time.perf_counter() - t0:.0f} s{' ' * 20}", flush=True)
    return solution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=float, default=1.0)
    ap.add_argument("--days", type=float, default=12.5)
    ap.add_argument("--spin-up", type=float, default=10.0)
    ap.add_argument("--internal-drag", type=float, default=1e-5)
    ap.add_argument("--constituent", default="M2")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=16)
    args = ap.parse_args()

    MEDIA.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)

    solution = build(args.resolution, args.days, args.spin_up,
                     args.internal_drag, args.constituent)
    con = args.constituent

    path = save_solution(solution, DATA / f"lte_{con.lower()}.npz")
    print(f"  solução -> {path}", flush=True)

    stats = solution.global_stats(con)
    amph = solution.find_amphidromes(con, 0.03)
    rot = solution.amphidrome_rotation_stats(con, amplitude_threshold=0.03)
    print(
        f"\n{con}: média {stats['amplitude_media']:.3f} m · "
        f"máx {stats['amplitude_maxima']:.2f} m · "
        f"oceano profundo {stats['amplitude_media_oceano_profundo']:.3f} m",
        flush=True,
    )
    print(f"{len(amph)} anfidromos:", flush=True)
    for hemisphere, s in rot.items():
        if s["total"]:
            print(f"  {hemisphere}: {s['total']} pontos, "
                  f"{100 * s['fracao_esperada']:.0f}% no sentido de Coriolis", flush=True)

    # --- validação -------------------------------------------------------
    rows = compare(solution, con, ALL_STATIONS)
    report = format_comparison(rows, con)
    print("\n" + report, flush=True)
    (DOCS / "lte_validation.txt").write_text(report + "\n")

    fig, _ = plot_validation(rows, con)
    fig.savefig(MEDIA / f"validacao_{con}.png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)

    # --- carta cotidal ---------------------------------------------------
    fig, _, _ = plot_cotidal_chart(solution, con, stations=OCEAN_STATIONS)
    fig.savefig(MEDIA / f"carta_cotidal_{con}.png", dpi=145,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nfiguras -> {MEDIA}", flush=True)

    # --- animações -------------------------------------------------------
    for label, builder, name in (
        ("onda: LTE contra equilíbrio", animate, "mare_onda"),
        ("correntes de maré", animate_currents, "mare_correntes"),
    ):
        out = MEDIA / f"{name}.mp4"
        print(f"[{label}] {args.frames} quadros a {args.fps} fps -> {out}", flush=True)
        t0 = time.perf_counter()
        anim, fig = builder(solution, constituent=con, frames=args.frames)
        _save_animation(anim, out, fps=args.fps, dpi=100)
        plt.close(fig)
        print(f"  {out.stat().st_size / 1e6:.1f} MB em "
              f"{time.perf_counter() - t0:.0f} s", flush=True)

    print(f"\nTudo pronto em {MEDIA}")


if __name__ == "__main__":
    main()
