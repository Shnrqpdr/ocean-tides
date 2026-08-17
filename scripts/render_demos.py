"""Gera os três vídeos de demonstração em ``media/``.

Cada animação usa uma janela de tempo diferente, porque cada uma revela um
fenômeno de escala temporal distinta:

* **polar** -- 6 dias, ~83 quadros/dia: a Terra girando sob os bojos. É a
  escala do dia lunar (24h50min), e precisa de muitos quadros por dia para a
  rotação não sair truncada.
* **mapa** -- 30 dias: o envelope de sizígia/quadratura, cujo batimento é de
  14.77 dias. Menos quadros por dia, mas cobre dois ciclos completos.
* **globo** -- 30 dias: o ciclo de declinação da Lua (27.3 dias), que faz os
  bojos oscilarem em latitude e produz a desigualdade diurna.

Uso: ``uv run python scripts/render_demos.py``
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem display, antes de importar pyplot

import matplotlib.pyplot as plt

from oceantides.cli import _save_animation
from oceantides.simulation import SimConfig, run_simulation
from oceantides.viz import globe3d, polar_slice, world_map

OUT = Path(__file__).resolve().parent.parent / "media"

JOBS = (
    # (nome, módulo, config, quadros, fps)
    ("mare_polar", polar_slice, SimConfig(days=6.0, dt=60.0, sample_dt=300.0), 500, 25),
    ("mare_mapa", world_map, SimConfig(days=30.0, dt=60.0, sample_dt=900.0), 750, 25),
    ("mare_globo", globe3d, SimConfig(days=30.0, dt=60.0, sample_dt=900.0), 560, 25),
)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cache: dict = {}

    for name, module, config, frames, fps in JOBS:
        key = (config.days, config.dt, config.sample_dt)
        if key not in cache:
            t0 = time.perf_counter()
            cache[key] = run_simulation(config)
            print(f"  simulados {config.days:g} dias em {time.perf_counter() - t0:.1f}s",
                  flush=True)

        path = OUT / f"{name}.mp4"
        print(f"[{name}] {frames} quadros -> {path}", flush=True)

        t0 = time.perf_counter()
        anim, fig = module.animate(cache[key], frames=frames)
        _save_animation(anim, path, fps, dpi=100)
        plt.close(fig)

        size_mb = path.stat().st_size / 1e6
        elapsed = time.perf_counter() - t0
        print(f"  pronto: {size_mb:.1f} MB, {frames / fps:.0f}s de vídeo, "
              f"renderizado em {elapsed:.0f}s", flush=True)

    print(f"\nTrês vídeos em {OUT}")


if __name__ == "__main__":
    main()
