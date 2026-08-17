"""A maré real contra a maré do livro, lado a lado.

Esta é a animação que justifica todo o módulo :mod:`oceantides.lte`. À
esquerda, a maré M2 que sai das Equações de Maré de Laplace; à direita, a maré
de equilíbrio da mesma constituinte -- o bojo que Newton previu, sempre em fase
com o potencial, sempre embaixo da Lua.

As duas usam **a mesma escala de cor**, de propósito. O contraste é o
argumento:

* o equilíbrio é um padrão liso de dois lobos, com no máximo ~24 cm;
* a maré real é irregular, chega a vários metros, e tem **buracos** -- os
  pontos anfidrômicos, onde a amplitude é zero e em torno dos quais a onda
  gira.

A razão é uma só, e é quantitativa: a onda de gravidade viaja a
``c = sqrt(gH) ~ 198 m/s`` num oceano de 4000 m, enquanto o ponto sublunar
varre o solo do equador a ~448 m/s. **A onda não consegue acompanhar a Lua.** O que
sobra não é um bojo atrasado, é um sistema de ondas refletidas e ressonantes.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ..constants import G_SURFACE, LOVE_FACTOR, OMEGA_EARTH, R_EARTH
from ..harmonic import CONSTITUENTS
from .common import PALETTE, SERIES, tide_cmap
from .cotidal import LAND, amplitude_cmap

__all__ = ["equilibrium_constituent_field", "animate", "animate_currents"]


def equilibrium_constituent_field(lat, lon, name, t):
    """Maré de equilíbrio de uma constituinte, na forma geodésica padrão."""
    con = CONSTITUENTS[name]
    phi = np.radians(np.asarray(lat))[:, None]
    lam = np.radians(np.asarray(lon))[None, :]

    if con.species == 0:
        shape = (3.0 * np.sin(phi) ** 2 - 1.0) / 2.0
    elif con.species == 1:
        shape = np.sin(2.0 * phi)
    else:
        shape = np.cos(phi) ** 2
    return (
        con.amplitude * LOVE_FACTOR * shape
        * np.cos(con.omega * t + con.species * lam)
    )


def _speed_note(solution):
    """A comparação numérica que explica tudo, calculada dos próprios dados.

    A velocidade relevante do ponto sublunar é a **relativa ao solo**, ou seja
    ``2 pi R / dia lunar``, e não ``Omega R`` -- esta última é a velocidade de
    um ponto do equador no espaço inercial, que não é com quem a onda compete.
    A diferença é de ~4% e não muda a conclusão, mas o número certo é 448 m/s.
    """
    from ..constants import T_LUNAR_DAY

    deep = solution.depth[solution.mask]
    c = float(np.sqrt(G_SURFACE * np.median(deep)))
    sublunar = float(2.0 * np.pi * R_EARTH / T_LUNAR_DAY)
    return (
        f"onda de maré: c = √(gH) = {c:.0f} m/s (H mediana {np.median(deep):.0f} m)"
        f"   ·   o ponto sublunar varre o equador a {sublunar:.0f} m/s"
        f"   →   a onda perde por {sublunar / c:.1f}×"
    )


def animate(solution, constituent="M2", frames=120, interval=60, **_):
    """Anima um ciclo completo da constituinte: LTE contra equilíbrio."""
    con = CONSTITUENTS[constituent]
    lon, lat = solution.lon, solution.lat

    amp, _ = solution.constituent(constituent)
    vmax = float(np.nanpercentile(amp, 99.0))

    fig, (ax_d, ax_e) = plt.subplots(
        1, 2, figsize=(15.0, 6.2), facecolor=PALETTE["page"]
    )
    fig.subplots_adjust(left=0.045, right=0.925, top=0.83, bottom=0.13, wspace=0.10)

    cmap = tide_cmap().with_extremes(bad=LAND)

    meshes = []
    for ax, title in (
        (ax_d, f"Maré dinâmica — Equações de Laplace"),
        (ax_e, f"Maré de equilíbrio — o bojo do livro"),
    ):
        ax.set_facecolor(LAND)
        m = ax.pcolormesh(
            lon, lat, np.ma.masked_invalid(np.zeros_like(amp)), cmap=cmap,
            vmin=-vmax, vmax=vmax, shading="gouraud", zorder=2,
        )
        meshes.append(m)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, color=PALETTE["ink"], fontsize=12, loc="left", pad=10)
    ax_e.tick_params(labelleft=False)

    amphidromes = solution.find_amphidromes(constituent, 0.03)
    if amphidromes:
        ax_d.plot([a[1] for a in amphidromes], [a[0] for a in amphidromes], "o",
                  markersize=5.5, markerfacecolor="none", markeredgecolor=PALETTE["ink"],
                  markeredgewidth=1.4, zorder=6)

    cbar = fig.colorbar(meshes[0], ax=[ax_d, ax_e], pad=0.012, fraction=0.026)
    cbar.set_label(f"elevação de {constituent}  [m]",
                   color=PALETTE["ink_secondary"], fontsize=9)
    cbar.ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)
    cbar.outline.set_visible(False)

    title = fig.text(0.045, 0.955, "", color=PALETTE["ink"], fontsize=16, weight="bold")
    sub = fig.text(0.045, 0.905, "", color=PALETTE["ink_secondary"], fontsize=10)
    fig.text(0.045, 0.035, _speed_note(solution),
             color=PALETTE["ink_muted"], fontsize=9)
    fig.text(
        0.045, 0.008,
        f"círculos = pontos anfidrômicos, onde a maré não sobe nem desce "
        f"({len(amphidromes)} encontrados)  ·  mesma escala de cor nos dois painéis",
        color=PALETTE["ink_muted"], fontsize=8.5,
    )

    # A maré de equilíbrio é ~12x menor que a dinâmica. Na mesma escala de cor
    # -- que é o ponto da figura -- ela fica quase invisível, então recebe
    # isolinhas próprias para a estrutura de dois lobos continuar legível sem
    # falsear a escala.
    eq_levels = np.array([-0.15, -0.10, -0.05, 0.05, 0.10, 0.15])
    contours = {"set": None}

    def update(k):
        t = k / frames * con.period
        dyn = solution.elevation_at(constituent, t)
        eq = np.where(solution.mask,
                      equilibrium_constituent_field(lat, lon, constituent, t), np.nan)

        meshes[0].set_array(np.ma.masked_invalid(dyn))
        meshes[1].set_array(np.ma.masked_invalid(eq))

        if contours["set"] is not None:
            contours["set"].remove()
        contours["set"] = ax_e.contour(
            lon, lat, np.ma.masked_invalid(eq), levels=eq_levels,
            colors=[PALETTE["ink_muted"]], linewidths=0.7, zorder=4,
        )

        hours = t / 3600.0
        title.set_text(
            f"{constituent}: onda real contra bojo teórico — "
            f"{hours:4.1f} h de um ciclo de {con.period_hours:.2f} h"
        )
        peak_dyn = max(abs(np.nanmin(dyn)), abs(np.nanmax(dyn)))
        peak_eq = max(abs(np.nanmin(eq)), abs(np.nanmax(eq)))
        sub.set_text(
            f"instantâneo: dinâmica de {np.nanmin(dyn):+.2f} a {np.nanmax(dyn):+.2f} m"
            f"   ·   equilíbrio de {np.nanmin(eq):+.2f} a {np.nanmax(eq):+.2f} m"
            f"   →   a real é {peak_dyn / max(peak_eq, 1e-9):.0f}× maior"
        )
        return (*meshes, title, sub)

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    return anim, fig


def animate_currents(solution, constituent="M2", frames=120, interval=60,
                     stride=4, **_):
    """Correntes de maré sobre o campo de velocidade máxima.

    Mostra o que a elevação esconde: nos pontos anfidrômicos a maré não sobe,
    mas a água **corre**. É por ali que a energia passa.
    """
    con = CONSTITUENTS[constituent]
    lon, lat = solution.lon, solution.lat
    speed = solution.current_amplitude(constituent)
    vmax = float(np.nanpercentile(speed, 99.0))

    fig, ax = plt.subplots(figsize=(13.5, 7.4), facecolor=PALETTE["page"])
    fig.subplots_adjust(left=0.055, right=0.90, top=0.86, bottom=0.09)
    ax.set_facecolor(LAND)

    mesh = ax.pcolormesh(lon, lat, np.ma.masked_invalid(speed), cmap=amplitude_cmap(),
                         vmin=0, vmax=vmax, shading="gouraud", zorder=2)

    # Só pontos de oceano recebem seta: desenhá-las em terra produz um
    # pontilhado que descaracteriza os continentes.
    sl = (slice(None, None, stride), slice(None, None, stride))
    lon_q, lat_q = np.meshgrid(lon, lat, indexing="xy")
    wet = solution.mask[sl]
    qx, qy = lon_q[sl][wet], lat_q[sl][wet]
    quiver = ax.quiver(
        qx, qy, np.zeros_like(qx), np.zeros_like(qy),
        color=PALETTE["ink"], scale=vmax * 55, width=0.0016, zorder=5, alpha=0.75,
    )

    amphidromes = solution.find_amphidromes(constituent, 0.03)
    if amphidromes:
        ax.plot([a[1] for a in amphidromes], [a[0] for a in amphidromes], "o",
                markersize=7, markerfacecolor="none", markeredgecolor=SERIES[1],
                markeredgewidth=1.7, zorder=6)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.032)
    cbar.set_label("velocidade máxima da corrente  [m/s]",
                   color=PALETTE["ink_secondary"], fontsize=9)
    cbar.ax.tick_params(colors=PALETTE["ink_muted"], labelsize=8.5)
    cbar.outline.set_visible(False)

    title = fig.text(0.055, 0.955, "", color=PALETTE["ink"], fontsize=16, weight="bold")
    fig.text(0.055, 0.915,
             "setas = corrente instantânea  ·  círculos = pontos anfidrômicos: "
             "a maré não sobe ali, mas a água corre",
             color=PALETTE["ink_secondary"], fontsize=10)

    def update(k):
        t = k / frames * con.period
        u, v = solution.current_at(constituent, t)
        quiver.set_UVC(np.nan_to_num(u[sl][wet]), np.nan_to_num(v[sl][wet]))
        title.set_text(
            f"Correntes de maré {constituent} — {t / 3600:4.1f} h "
            f"de {con.period_hours:.2f} h"
        )
        return quiver, title

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    return anim, fig
