"""Mapa global animado da maré, com painel de marégrafos.

Painel superior: ``h(lat, lon)`` num mapa plano, em escala **divergente**
ancorada em zero -- vermelho acima do nível médio, azul abaixo, cinza no nível
médio. Os dois bojos aparecem como duas manchas vermelhas em lados opostos do
planeta, e a Terra gira sob elas: é a explicação do PDF para as duas marés
altas por dia, vista de fora.

Painel inferior: séries temporais de quatro estações escolhidas pelo contraste
de resposta -- de uma bacia quase ressonante a uma que responde *abaixo* do
equilíbrio. Ao longo de 30 dias o envelope de sizígia e quadratura aparece
sozinho, com o batimento de 14.8 dias.

O campo global **não** é lido de disco: é recalculado por quadro a partir das
efemérides. É forma fechada e custa ~2 ms; armazenar 4000 quadros numa grade
181x361 daria ~1 GB sem nenhum ganho.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ..grid import Grid, subbody_point
from .common import (
    PALETTE,
    SERIES,
    frame_indices,
    style_axes,
    tide_cmap,
    time_label,
)

# Quatro estações que cobrem o espectro de resposta, na ordem categórica fixa.
FEATURED = ("Burntcoat Head", "Sao Luis", "Rio de Janeiro", "Papeete")


def _select_stations(result, names):
    lookup = {n.lower(): i for i, n in enumerate(result.station_names)}
    return [(n, lookup[n.lower()]) for n in names if n.lower() in lookup]


def animate(result, exaggeration=None, frames=None, interval=40, stations=FEATURED):
    idx = frame_indices(result.t.size, frames)
    grid = Grid(*result.config.grid_shape)
    bodies = result.rebuild_bodies()
    chosen = _select_stations(result, stations)

    # Escala de cor fixada em toda a série, simétrica em torno de zero.
    peak = 0.0
    for i in idx[:: max(1, len(idx) // 40)]:
        peak = max(peak, float(np.abs(
            grid.equilibrium_field(bodies, result.t[i], result.config.theta0,
                                   result.config.love)
        ).max()))
    peak = max(peak, 1e-6)

    fig = plt.figure(figsize=(13.0, 9.8), facecolor=PALETTE["page"])
    # Small multiples em vez de um painel único: com Fundy em +-3 m e Papeete
    # em +-0.2 m, as estações pequenas somem num eixo comum -- e eixo duplo
    # nunca é a resposta. Cada bacia ganha seu próprio eixo y.
    gs = fig.add_gridspec(
        1 + len(chosen), 1,
        height_ratios=[3.0] + [0.72] * len(chosen),
        hspace=0.18, left=0.075, right=0.885, top=0.90, bottom=0.075,
    )
    ax_map = fig.add_subplot(gs[0])
    ts_axes = [fig.add_subplot(gs[1])]
    ts_axes += [fig.add_subplot(gs[j + 2], sharex=ts_axes[0]) for j in range(len(chosen) - 1)]

    # ---------------- mapa ----------------
    ax_map.set_facecolor(PALETTE["surface"])
    field = grid.equilibrium_field(bodies, result.t[idx[0]], result.config.theta0,
                                   result.config.love)
    mesh = ax_map.pcolormesh(
        grid.lon, grid.lat, field, cmap=tide_cmap(), vmin=-peak, vmax=peak,
        shading="gouraud",
    )
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-90, 90)
    ax_map.set_xticks(range(-180, 181, 60))
    ax_map.set_yticks(range(-90, 91, 30))
    ax_map.set_xlabel("longitude", color=PALETTE["ink_secondary"], fontsize=9)
    ax_map.set_ylabel("latitude", color=PALETTE["ink_secondary"], fontsize=9)
    style_axes(ax_map, grid=False, spines=())
    ax_map.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)

    _add_coastlines(ax_map)

    cbar = fig.colorbar(mesh, ax=ax_map, pad=0.015, fraction=0.045)
    cbar.set_label("altura da maré de equilíbrio  [m]",
                   color=PALETTE["ink_secondary"], fontsize=9)
    cbar.ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)
    cbar.outline.set_visible(False)

    (sub_moon,) = ax_map.plot([], [], "o", markersize=9, markerfacecolor="none",
                              markeredgecolor=PALETTE["ink"], markeredgewidth=1.8,
                              zorder=6)
    moon_txt = ax_map.text(0, 0, "", color=PALETTE["ink"], fontsize=9,
                           ha="center", va="bottom", zorder=7)
    # Marcadores dos corpos usam tinta e formas distintas, nunca cores de série:
    # laranja já identifica uma estação, e reusá-la aqui criaria ambiguidade.
    (sub_sun,) = ax_map.plot([], [], "s", markersize=8, markerfacecolor="none",
                             markeredgecolor=PALETTE["ink_secondary"],
                             markeredgewidth=1.8, zorder=6)
    sun_txt = ax_map.text(0, 0, "", color=PALETTE["ink_secondary"], fontsize=9,
                          ha="center", va="bottom", zorder=7)

    # marcadores das estações destacadas, com rótulo direto (regra de relevo)
    from ..stations import get_stations

    all_stations = get_stations(result.config.stations)
    for slot, (name, i) in enumerate(chosen):
        st = all_stations[i]
        ax_map.plot(st.lon, st.lat, "o", markersize=6, color=SERIES[slot],
                    markeredgecolor=PALETTE["surface"], markeredgewidth=1.4, zorder=8)
        ax_map.annotate(
            st.name, (st.lon, st.lat), textcoords="offset points", xytext=(7, -3),
            color=PALETTE["ink_secondary"], fontsize=8, zorder=8,
        )

    # ---------------- séries temporais (small multiples) ----------------
    days = result.days
    cursors = []
    for slot, ((name, i), ax) in enumerate(zip(chosen, ts_axes)):
        st = all_stations[i]
        series = result.h_dynamic[:, i]
        ax.plot(days, series, color=SERIES[slot], linewidth=1.4)
        ax.axhline(0.0, color=PALETTE["axis"], linewidth=0.9, zorder=1)

        # Folga no topo do eixo para o cabeçalho, em vez de mais altura de figura.
        peak_y = max(np.abs(series).max(), 1e-3)
        ax.set_ylim(-peak_y * 1.22, peak_y * 2.05)
        ax.set_xlim(days[0], days[-1])
        style_axes(ax, spines=("bottom",))
        tick = round(peak_y, 1) or round(peak_y, 2)
        ax.set_yticks([-tick, 0, tick])

        # A cor fica no marcador; o texto usa tinta. Assim a identidade nunca
        # depende só da cor -- exigido porque o aqua fica abaixo de 3:1.
        ax.annotate("●", (0.005, 0.97), xycoords="axes fraction",
                    color=SERIES[slot], fontsize=10, va="top")
        ax.annotate(st.name, (0.022, 0.965), xycoords="axes fraction",
                    color=PALETTE["ink"], fontsize=9.5, va="top", weight="bold")
        ax.annotate(
            f"amplitude {np.ptp(series):.2f} m   ·   amplificação "
            f"{result.meta['m2_amplification'][i]:.1f}×   ·   atraso "
            f"{result.meta['m2_lag_hours'][i]:.1f} h   ·   observada "
            f"{st.observed_range:.1f} m",
            (0.998, 0.965), xycoords="axes fraction", ha="right",
            color=PALETTE["ink_muted"], fontsize=8, va="top",
        )

        cursors.append(ax.axvline(days[idx[0]], color=PALETTE["ink"],
                                  linewidth=1.1, alpha=0.5, zorder=9))
        _mark_spring_neap(ax, result, label=(slot == 0))

        if slot < len(chosen) - 1:
            ax.tick_params(labelbottom=False)

    ts_axes[-1].set_xlabel("dias desde a lua nova",
                           color=PALETTE["ink_secondary"], fontsize=9)
    ts_axes[len(chosen) // 2].set_ylabel("nível [m]",
                                         color=PALETTE["ink_secondary"], fontsize=9)

    title = fig.text(0.075, 0.962, "", color=PALETTE["ink"], fontsize=15, weight="bold")
    subtitle = fig.text(0.075, 0.936, "", color=PALETTE["ink_secondary"], fontsize=10)
    fig.text(
        0.885, 0.012,
        "campo global = maré de equilíbrio (Eq. 5.55)  ·  séries = resposta dinâmica de "
        "bacia, cada painel com sua própria escala",
        color=PALETTE["ink_muted"], fontsize=8.5, ha="right",
    )

    def update(k):
        i = idx[k]
        t = result.t[i]
        field = grid.equilibrium_field(bodies, t, result.config.theta0,
                                       result.config.love)
        mesh.set_array(field)

        if "moon" in result.config.bodies:
            lat, lon = subbody_point(result.moon_pos[i], t, result.config.theta0)
            sub_moon.set_data([lon], [lat])
            moon_txt.set_position((float(lon), float(lat) + 4))
            moon_txt.set_text("ponto sublunar")
        if "sun" in result.config.bodies:
            lat, lon = subbody_point(result.sun_pos[i], t, result.config.theta0)
            sub_sun.set_data([lon], [lat])
            sun_txt.set_position((float(lon), float(lat) + 4))
            sun_txt.set_text("subsolar")

        for cur in cursors:
            cur.set_xdata([days[i], days[i]])

        title.set_text(f"Maré global — {time_label(t)}")
        subtitle.set_text(
            f"amplitude instantânea do campo {np.ptp(field):.2f} m"
            f"  ·  média sobre a esfera {grid.area_mean(field):+.0e} m (a maré não cria água)"
        )
        return (mesh, sub_moon, sub_sun, title, subtitle, *cursors)

    anim = FuncAnimation(fig, update, frames=len(idx), interval=interval, blit=False)
    return anim, fig


def _mark_spring_neap(ax, result, label=True):
    """Assinala sizígias e quadraturas pelo ângulo Sol-Lua.

    O padding com infinito faz com que extremos *na borda* também sejam
    detectados -- importante porque a simulação começa exatamente na lua nova,
    ou seja, numa sizígia em ``t = 0``.
    """
    if not {"moon", "sun"} <= set(result.config.bodies):
        return
    m = result.moon_pos / np.linalg.norm(result.moon_pos, axis=1, keepdims=True)
    s = result.sun_pos / np.linalg.norm(result.sun_pos, axis=1, keepdims=True)
    sep = np.degrees(np.arccos(np.clip(np.sum(m * s, axis=1), -1, 1)))
    aligned = np.minimum(sep, 180 - sep)  # 0 = sizígia, 90 = quadratura

    for value, name, style in (
        (aligned, "sizígia", (0, (2, 3))),
        (-aligned, "quadratura", (0, (1, 4))),
    ):
        padded = np.r_[np.inf, value, np.inf]
        is_min = (padded[1:-1] <= padded[:-2]) & (padded[1:-1] < padded[2:])
        hits = np.where(is_min)[0]
        for rank, j in enumerate(hits):
            ax.axvline(result.days[j], color=PALETTE["ink_muted"], linewidth=0.9,
                       linestyle=style, zorder=2)
            # rotula só a primeira ocorrência; as tracejadas seguintes se explicam
            if label and rank == 0 and result.days[j] < result.days[-1] * 0.8:
                ax.annotate(
                    name, (result.days[j], 0.02), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(4, 0), va="bottom",
                    color=PALETTE["ink_muted"], fontsize=8,
                )


def _add_coastlines(ax):
    """Linhas de costa via cartopy, se instalado. Silencioso se não estiver."""
    try:
        import cartopy.feature as cfeature
        from cartopy.mpl.geoaxes import GeoAxes  # noqa: F401
    except ImportError:
        return
    try:
        import cartopy.io.shapereader as shpreader
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        reader = shpreader.Reader(
            shpreader.natural_earth(resolution="110m", category="physical",
                                    name="coastline")
        )
        for geom in reader.geometries():
            for line in getattr(geom, "geoms", [geom]):
                xs, ys = np.array(line.coords).T
                ax.plot(xs, ys, color=PALETTE["ink"], linewidth=0.5, alpha=0.35,
                        zorder=5)
    except Exception:
        pass
