"""Carta cotidal: a figura clássica da maré, e a prova de que ela é uma onda.

Duas famílias de curvas, sobrepostas ao globo:

* **linhas de igual amplitude** (co-range), em escala sequencial de um só tom --
  amplitude é magnitude, não polaridade, então não leva escala divergente;
* **linhas de igual fase** (cotidais), em tinta, marcando o instante em que a
  maré alta chega. Cada linha é uma frente de onda.

O que salta aos olhos são os pontos onde **todas as linhas cotidais convergem**
e a amplitude vai a zero: os **pontos anfidrômicos**. Ali a maré simplesmente
não sobe nem desce, e a fase percorre 360 graus ao redor. São a assinatura
inconfundível de uma onda girando sob a força de Coriolis dentro de uma bacia.

A teoria de equilíbrio não tem como produzi-los: ela prevê maré máxima sob a
Lua, em toda parte, sempre em fase com o potencial. Comparar esta figura com o
mapa da maré de equilíbrio é o argumento visual mais direto de que o oceano
real responde como onda forçada, e não como bojo hidrostático.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .common import PALETTE, SERIES

__all__ = ["amplitude_cmap", "plot_cotidal_chart", "plot_validation"]

# Rampa sequencial azul, clara -> escura. Amplitude é magnitude: um só tom,
# nunca arco-íris, e o passo mais claro pode recuar para o fundo.
_BLUE_RAMP = [
    "#eaf3fd", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
    "#104281", "#0d366b",
]

LAND = "#e4e2dc"


def amplitude_cmap():
    return LinearSegmentedColormap.from_list("amplitude", _BLUE_RAMP).with_extremes(
        bad=LAND
    )


def _cotidal_lines(ax, lon, lat, phase, levels, color, linewidth=0.7):
    """Desenha linhas de fase constante lidando com a descontinuidade 0/360.

    Contornar diretamente um campo cíclico cria uma linha espúria onde ele
    salta de 359 para 0 graus. A saída é, para cada nível, contornar a
    **diferença angular com sinal** e mascarar a região distante do nível,
    que é justamente onde o salto vive.
    """
    drawn = []
    for level in levels:
        diff = ((phase - level + 180.0) % 360.0) - 180.0
        masked = np.ma.masked_invalid(np.where(np.abs(diff) > 90.0, np.nan, diff))
        if masked.count() == 0:
            continue
        cs = ax.contour(lon, lat, masked, levels=[0.0], colors=[color],
                        linewidths=linewidth, zorder=4)
        drawn.append((level, cs))
    return drawn


def plot_cotidal_chart(solution, constituent="M2", phase_step=30,
                       amphidrome_threshold=0.03, stations=None, figsize=(13.5, 7.6)):
    """Carta cotidal completa de uma constituinte."""
    amp, phase = solution.constituent(constituent)
    lon, lat = solution.lon, solution.lat

    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["page"])
    fig.subplots_adjust(left=0.055, right=0.90, top=0.86, bottom=0.09)
    ax.set_facecolor(LAND)

    vmax = float(np.nanpercentile(amp, 99.5))
    mesh = ax.pcolormesh(
        lon, lat, np.ma.masked_invalid(amp), cmap=amplitude_cmap(),
        vmin=0.0, vmax=vmax, shading="gouraud", zorder=2,
    )

    levels = list(range(0, 360, phase_step))
    _cotidal_lines(ax, lon, lat, phase, levels, PALETTE["ink"], 0.65)

    amphidromes = solution.find_amphidromes(constituent, amphidrome_threshold)
    if amphidromes:
        alat = [a[0] for a in amphidromes]
        alon = [a[1] for a in amphidromes]
        ax.plot(alon, alat, "o", markersize=7, markerfacecolor="none",
                markeredgecolor=SERIES[1], markeredgewidth=1.7, zorder=6)

    if stations:
        for st in stations:
            ax.plot(st.lon, st.lat, "^", markersize=6, color=SERIES[2],
                    markeredgecolor=PALETTE["surface"], markeredgewidth=1.0, zorder=7)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("longitude", color=PALETTE["ink_secondary"], fontsize=9)
    ax.set_ylabel("latitude", color=PALETTE["ink_secondary"], fontsize=9)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.032)
    cbar.set_label(f"amplitude de {constituent}  [m]",
                   color=PALETTE["ink_secondary"], fontsize=9)
    cbar.ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)
    cbar.outline.set_visible(False)

    fig.text(0.055, 0.955, f"Carta cotidal de {constituent}",
             color=PALETTE["ink"], fontsize=16, weight="bold")
    fig.text(
        0.055, 0.915,
        f"cores = amplitude  ·  linhas finas = fase a cada {phase_step}° "
        f"(frentes de onda)  ·  círculos = {len(amphidromes)} pontos anfidrômicos",
        color=PALETTE["ink_secondary"], fontsize=10,
    )
    stats = solution.global_stats(constituent)
    fig.text(
        0.90, 0.025,
        f"amplitude média {stats['amplitude_media']:.3f} m  ·  "
        f"máxima {stats['amplitude_maxima']:.2f} m  ·  "
        f"oceano profundo {stats['amplitude_media_oceano_profundo']:.3f} m  ·  "
        f"grade {solution.mask.shape[0]}×{solution.mask.shape[1]}",
        color=PALETTE["ink_muted"], fontsize=8.5, ha="right",
    )
    return fig, ax, amphidromes


def plot_validation(rows, constituent="M2", figsize=(8.4, 7.4)):
    """Dispersão modelo vs. observação, separando ilhas de estações costeiras.

    A separação é o ponto: um modelo global grosseiro tem chance real nas
    ilhas, que amostram oceano aberto, e nenhuma na costa, onde a plataforma
    não existe na grade.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["page"])
    fig.subplots_adjust(left=0.12, right=0.96, top=0.87, bottom=0.10)
    ax.set_facecolor(PALETTE["surface"])

    # Escala logarítmica: as amplitudes vão de ~0.12 m a 3.5 m, e num eixo
    # linear todas as ilhas colapsam num aglomerado ilegível no canto.
    groups = {"ilha": SERIES[0], "costeira": SERIES[1]}
    lo, hi = np.inf, 0.0
    labelled = []
    for kind, color in groups.items():
        sel = [(o, m, st.name) for st, o, m, _ in rows
               if st.kind == kind and np.isfinite(m) and o > 0 and m > 0]
        if not sel:
            continue
        obs = np.array([s[0] for s in sel])
        mod = np.array([s[1] for s in sel])
        lo = min(lo, obs.min(), mod.min())
        hi = max(hi, obs.max(), mod.max())
        rms = np.sqrt(np.mean((mod - obs) ** 2)) * 100
        ax.plot(obs, mod, "o", color=color, markersize=8,
                markeredgecolor=PALETTE["surface"], markeredgewidth=1.2,
                label=f"{kind} — RMS {rms:.1f} cm  (n={len(sel)})", zorder=5)
        labelled += sel

    lo, hi = lo / 1.7, hi * 1.7
    ax.set_xscale("log")
    ax.set_yscale("log")

    # rótulos alternando acima/abaixo, para o aglomerado não se sobrepor
    for k, (o, m, name) in enumerate(sorted(labelled)):
        dy = 9 if k % 2 == 0 else -13
        ax.annotate(name.split(",")[0], (o, m), textcoords="offset points",
                    xytext=(8, dy), fontsize=7.5, color=PALETTE["ink_muted"])

    ax.plot([lo, hi], [lo, hi], color=PALETTE["axis"], linewidth=1.2,
            linestyle=(0, (4, 3)), zorder=2)
    for factor, label in ((2.0, "2× acima"), (0.5, "2× abaixo")):
        ax.plot([lo, hi], [lo * factor, hi * factor], color=PALETTE["grid"],
                linewidth=1.0, zorder=1)

    ax.annotate("concordância perfeita", (hi * 0.36, hi * 0.36),
                textcoords="offset points", xytext=(4, -13), rotation=45,
                fontsize=8, color=PALETTE["ink_muted"])
    ax.annotate("fator 2", (hi * 0.36, hi * 0.18), textcoords="offset points",
                xytext=(4, -12), rotation=45, fontsize=7.5,
                color=PALETTE["ink_muted"])

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"amplitude observada de {constituent}  [m]  (marégrafo NOAA)",
                  color=PALETTE["ink_secondary"], fontsize=9.5)
    ax.set_ylabel(f"amplitude modelada de {constituent}  [m]  (LTE)",
                  color=PALETTE["ink_secondary"], fontsize=9.5)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(PALETTE["axis"])
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=9)

    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    fig.text(0.12, 0.95, "Validação contra marégrafos", color=PALETTE["ink"],
             fontsize=15, weight="bold")
    fig.text(0.12, 0.915,
             "ilhas amostram oceano aberto; estações costeiras exigem plataforma "
             "que a grade não resolve",
             color=PALETTE["ink_secondary"], fontsize=9.5)
    return fig, ax
