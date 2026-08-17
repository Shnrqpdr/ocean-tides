"""Globo 3D deformado -- as Figuras 5-11b e 5-13 em três dimensões.

A superfície é uma esfera cujo raio é modulado pela maré (muito exagerada) e
colorida pela mesma escala divergente do mapa. Ao contrário do corte
equatorial, aqui a **inclinação do bojo em latitude** fica visível: quando a
Lua está em declinação alta, os dois bojos não são simétricos em relação ao
equador, e um ponto na superfície passa por uma maré alta forte e outra fraca
a cada dia. É a *desigualdade diurna* que o PDF menciona na p. 204.

A grade é deliberadamente grossa (~60x120): ``plot_surface`` reconstrói a malha
inteira a cada quadro, e resolução alta derruba a taxa de quadros sem revelar
nada -- a maré é um harmônico de grau 2, extremamente suave.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ..constants import R_EARTH
from ..grid import Grid, subbody_point
from .common import PALETTE, auto_exaggeration, frame_indices, tide_cmap, time_label

N_LAT, N_LON = 61, 121


def animate(result, exaggeration=None, frames=None, interval=40, elev=18.0):
    idx = frame_indices(result.t.size, frames)
    grid = Grid(N_LAT, N_LON)
    bodies = result.rebuild_bodies()

    peak = 0.0
    for i in idx[:: max(1, len(idx) // 40)]:
        peak = max(peak, float(np.abs(
            grid.equilibrium_field(bodies, result.t[i], result.config.theta0,
                                   result.config.love)
        ).max()))
    peak = max(peak, 1e-6)

    factor, exag_text = auto_exaggeration(peak, R_EARTH, target_fraction=0.16)
    if exaggeration is not None:
        factor = exaggeration
        exag_text = f"bojo exagerado {factor:,.0f}×"

    lat = np.radians(grid.lat_mesh)
    lon = np.radians(grid.lon_mesh)
    cos_lat = np.cos(lat)
    base = np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)])

    fig = plt.figure(figsize=(9.6, 8.0), facecolor=PALETTE["page"])
    ax = fig.add_subplot(111, projection="3d")
    # Painéis e fundo transparentes: com os eixos desligados, o retângulo
    # branco do axes 3D só criaria uma moldura sem função.
    ax.patch.set_alpha(0.0)
    fig.subplots_adjust(left=0.02, right=0.84, top=0.90, bottom=0.03)

    cmap = tide_cmap()
    lim = 1.0 + factor * peak / R_EARTH * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    # zoom compensa o preenchimento generoso que o axes 3D reserva por padrão
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    ax.set_axis_off()

    # marcador do ponto sublunar, projetado sobre a superfície deformada
    (moon_marker,) = ax.plot([], [], [], "o", color=PALETTE["ink"], markersize=8,
                             zorder=10)

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=-peak, vmax=peak)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.0, fraction=0.03,
                        location="right")
    cbar.set_label("altura da maré de equilíbrio  [m]",
                   color=PALETTE["ink_secondary"], fontsize=9)
    cbar.ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)
    cbar.outline.set_visible(False)

    title = fig.text(0.045, 0.955, "", color=PALETTE["ink"], fontsize=15, weight="bold")
    subtitle = fig.text(0.045, 0.921, "", color=PALETTE["ink_secondary"], fontsize=10)
    fig.text(0.97, 0.02, exag_text, color=PALETTE["ink_muted"], fontsize=8.5, ha="right")

    state = {"surface": None}

    def update(k):
        i = idx[k]
        t = result.t[i]
        field = grid.equilibrium_field(bodies, t, result.config.theta0,
                                       result.config.love)

        radius = 1.0 + factor * field / R_EARTH
        x, y, z = base * radius

        if state["surface"] is not None:
            state["surface"].remove()
        state["surface"] = ax.plot_surface(
            x, y, z, facecolors=cmap(norm(field)), rstride=1, cstride=1,
            linewidth=0, antialiased=False, shade=False,
        )

        # a câmera acompanha a rotação da Terra, então o observador vê o bojo
        # passar -- e não a Terra parada com o bojo girando
        ax.view_init(elev=elev, azim=(-t / 86164.0905 * 360.0) % 360.0)

        declination = 0.0
        if "moon" in result.config.bodies:
            mlat, mlon = subbody_point(result.moon_pos[i], t, result.config.theta0)
            declination = float(mlat)
            j = int(np.argmin(np.abs(grid.lat - mlat)))
            m = int(np.argmin(np.abs(grid.lon - mlon)))
            moon_marker.set_data([x[j, m] * 1.04], [y[j, m] * 1.04])
            moon_marker.set_3d_properties([z[j, m] * 1.04])

        title.set_text(f"Bojo de maré em 3D — {time_label(t)}")
        subtitle.set_text(
            f"amplitude do campo {np.ptp(field):.2f} m"
            f"  ·  declinação da Lua {declination:+.1f}°"
            f"  ({'bojos assimétricos → desigualdade diurna' if abs(declination) > 10 else 'bojos simétricos'})"
        )
        return state["surface"], moon_marker, title, subtitle

    anim = FuncAnimation(fig, update, frames=len(idx), interval=interval, blit=False)
    return anim, fig
