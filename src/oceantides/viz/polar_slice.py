"""Corte equatorial animado -- as Figuras 5-10b, 5-11a/b e 5-13 ganhando vida.

Painel esquerdo: o **campo de força de maré** das Eq. 5.54, com as setas
apontando para fora ao longo do eixo Terra-Lua e para dentro em quadratura,
exatamente como na Figura 5-11a.

Painel direito: os **dois bojos**. O contorno tracejado é a maré de equilíbrio,
alinhada com a Lua; o contorno cheio é a resposta dinâmica de uma bacia,
amplificada e *girada à frente* pelo atraso de fase -- a Figura 5-13, em que a
maré alta não está sobre o eixo Terra-Lua.

Escala: a Lua está a 60.3 raios terrestres, longe demais para caber junto com
uma Terra visível, então o raio orbital é comprimido e isso vai rotulado.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ..constants import GM_MOON, GM_SUN, R_EARTH, T_M2
from ..forcing import equilibrium_height, tidal_accel_expanded
from ..grid import eci_to_ecef, greenwich_angle
from ..response import steady_state_response
from .common import (
    PALETTE,
    SERIES,
    auto_exaggeration,
    frame_indices,
    phase_lag_to_angle,
    style_axes,
    time_label,
)

MOON_DRAW_RADIUS = 2.15  # em raios terrestres, comprimido para caber
N_THETA = 361
N_ARROWS = 24


def _equatorial_ring(n=N_THETA, radius=R_EARTH):
    theta = np.linspace(0.0, 2.0 * np.pi, n)
    pts = radius * np.stack(
        [np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1
    )
    return theta, pts


def _bodies_in_equatorial_plane(result, i):
    """Projeta Lua e Sol no plano equatorial, no referencial inercial."""
    out = {}
    if "moon" in result.config.bodies:
        out["moon"] = (result.moon_pos[i], GM_MOON)
    if "sun" in result.config.bodies:
        out["sun"] = (result.sun_pos[i], GM_SUN)
    return out


def animate(result, exaggeration=None, frames=None, interval=40,
            basin_period_h=12.9, basin_q=8.0):
    """Monta a animação do corte equatorial.

    ``basin_period_h`` e ``basin_q`` definem a bacia de referência usada para
    desenhar o bojo dinâmico atrasado do painel direito.
    """
    idx = frame_indices(result.t.size, frames)
    theta, ring = _equatorial_ring()

    # Resposta da bacia de referência em M2 -> amplificação e giro do bojo.
    amp, lag = steady_state_response(2 * np.pi / (basin_period_h * 3600.0), basin_q,
                                     2 * np.pi / T_M2)
    amp, lag = float(amp), float(lag)
    bulge_shift = phase_lag_to_angle(lag)

    # Escala do exagero, fixada pelo maior bojo de toda a série.
    h_peak = 0.0
    for i in idx[:: max(1, len(idx) // 60)]:
        h = np.zeros(N_THETA)
        for pos, gm in _bodies_in_equatorial_plane(result, i).values():
            h += equilibrium_height(ring, pos, gm, love=result.config.love)
        h_peak = max(h_peak, float(np.abs(h).max()))
    h_peak = max(h_peak * amp, 1e-9)
    factor, exag_text = auto_exaggeration(h_peak, R_EARTH)
    if exaggeration is not None:
        factor = exaggeration
        exag_text = f"bojo exagerado {factor:,.0f}x"

    fig, (ax_f, ax_b) = plt.subplots(
        1, 2, figsize=(13.0, 6.6), facecolor=PALETTE["page"]
    )
    fig.subplots_adjust(left=0.04, right=0.97, top=0.86, bottom=0.10, wspace=0.12)

    for ax in (ax_f, ax_b):
        ax.set_facecolor(PALETTE["surface"])
        ax.set_aspect("equal")
        ax.set_xlim(-3.05, 3.05)
        ax.set_ylim(-2.15, 2.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    unit = np.stack([np.cos(theta), np.sin(theta)], axis=-1)

    # --- painel esquerdo: campo de forças (Fig. 5-11a) ----------------------
    ax_f.add_patch(plt.Circle((0, 0), 1.0, facecolor="#eef2f7",
                              edgecolor=PALETTE["axis"], linewidth=1.2, zorder=1))
    arrow_theta = np.linspace(0.0, 2.0 * np.pi, N_ARROWS, endpoint=False)
    arrow_pts = R_EARTH * np.stack(
        [np.cos(arrow_theta), np.sin(arrow_theta), np.zeros_like(arrow_theta)], axis=-1
    )
    quiver = ax_f.quiver(
        np.cos(arrow_theta), np.sin(arrow_theta),
        np.zeros(N_ARROWS), np.zeros(N_ARROWS),
        color=SERIES[0], scale=1.0, scale_units="xy", width=0.006,
        zorder=4, angles="xy",
    )
    ax_f.set_title("Campo de força de maré  (Eq. 5.54)",
                   color=PALETTE["ink"], fontsize=12, pad=12, loc="left")

    # --- painel direito: bojos (Fig. 5-11b e 5-13) --------------------------
    ax_b.add_patch(plt.Circle((0, 0), 1.0, facecolor="#eef2f7",
                              edgecolor="none", zorder=2))
    ax_b.plot(unit[:, 0], unit[:, 1], color=PALETTE["axis"],
              linewidth=1.0, linestyle=(0, (2, 3)), zorder=3)
    (eq_line,) = ax_b.plot([], [], color=PALETTE["ink_muted"], linewidth=1.6,
                           linestyle=(0, (5, 3)), zorder=5,
                           label="maré de equilíbrio (Eq. 5.55)")
    (dyn_line,) = ax_b.plot([], [], color=SERIES[0], linewidth=2.2, zorder=6,
                            label=f"resposta dinâmica (T₀={basin_period_h:g} h, Q={basin_q:g})")
    (meridian,) = ax_b.plot([], [], color=PALETTE["ink_secondary"], linewidth=1.6,
                            zorder=7, label="meridiano de Greenwich")
    ax_b.set_title("Bojos oceânicos e o atraso da Fig. 5-13",
                   color=PALETTE["ink"], fontsize=12, pad=12, loc="left")

    # Lua e Sol, com raio comprimido
    moon_dot = ax_b.plot([], [], "o", color=PALETTE["ink"], markersize=9, zorder=8)[0]
    moon_dot_f = ax_f.plot([], [], "o", color=PALETTE["ink"], markersize=9, zorder=8)[0]
    moon_txt = ax_b.text(0, 0, "", color=PALETTE["ink_secondary"], fontsize=9,
                         ha="center", va="center")
    sun_arrow = ax_b.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=SERIES[1], linewidth=2.0),
    )
    sun_txt = ax_b.text(0, 0, "", color=PALETTE["ink_secondary"], fontsize=9,
                        ha="center", va="center")

    for ax in (ax_f, ax_b):
        ax.add_patch(plt.Circle((0, 0), MOON_DRAW_RADIUS, fill=False,
                                edgecolor=PALETTE["grid"], linewidth=0.9, zorder=0))

    legend = ax_b.legend(loc="lower left", frameon=False, fontsize=9,
                         bbox_to_anchor=(-0.02, -0.10))
    for text in legend.get_texts():
        text.set_color(PALETTE["ink_secondary"])

    title = fig.text(0.04, 0.955, "", color=PALETTE["ink"], fontsize=15, weight="bold")
    subtitle = fig.text(0.04, 0.915, "", color=PALETTE["ink_secondary"], fontsize=10)
    fig.text(
        0.97, 0.02,
        f"{exag_text}   ·   distância Terra–Lua comprimida (real: 60.3 raios terrestres)"
        f"   ·   amplificação {amp:.1f}×, atraso {lag / (2 * np.pi) * T_M2 / 3600:.1f} h",
        color=PALETTE["ink_muted"], fontsize=8.5, ha="right",
    )

    def update(k):
        i = idx[k]
        t = result.t[i]
        bodies = _bodies_in_equatorial_plane(result, i)

        # ---- forças (Eq. 5.54), normalizadas para caber no desenho
        accel = np.zeros((N_ARROWS, 3))
        for pos, gm in bodies.values():
            accel += tidal_accel_expanded(arrow_pts, pos, gm)
        scale = 0.55 / max(np.abs(accel[:, :2]).max(), 1e-30)
        quiver.set_UVC(accel[:, 0] * scale, accel[:, 1] * scale)

        # ---- alturas de equilíbrio ao longo do anel
        h = np.zeros(N_THETA)
        for pos, gm in bodies.values():
            h += equilibrium_height(ring, pos, gm, love=result.config.love)

        r_eq = 1.0 + factor * h / R_EARTH
        eq_line.set_data(r_eq * unit[:, 0], r_eq * unit[:, 1])

        # O bojo dinâmico é o de equilíbrio amplificado e girado para a frente
        # pelo atraso de fase -- ver phase_lag_to_angle.
        shift = int(round(bulge_shift / (2 * np.pi) * (N_THETA - 1)))
        r_dyn = 1.0 + factor * amp * np.roll(h, shift) / R_EARTH
        dyn_line.set_data(r_dyn * unit[:, 0], r_dyn * unit[:, 1])

        # ---- meridiano de Greenwich, mostrando a rotação da Terra
        ang = float(greenwich_angle(t)) % (2 * np.pi)
        meridian.set_data([0, np.cos(ang) * 0.97], [0, np.sin(ang) * 0.97])

        # ---- Lua e Sol
        if "moon" in bodies:
            mp = bodies["moon"][0]
            d = np.hypot(mp[0], mp[1])
            mx, my = MOON_DRAW_RADIUS * mp[0] / d, MOON_DRAW_RADIUS * mp[1] / d
            moon_dot.set_data([mx], [my])
            moon_dot_f.set_data([mx], [my])
            # rótulo para DENTRO da órbita, no vazio entre a Terra e a Lua:
            # para fora ele sairia dos limites do eixo e escreveria sobre o título
            moon_txt.set_position((mx / MOON_DRAW_RADIUS * 1.78,
                                   my / MOON_DRAW_RADIUS * 1.78))
            moon_txt.set_text("Lua")

        phase_txt = ""
        if "sun" in bodies and "moon" in bodies:
            sp, mp = bodies["sun"][0], bodies["moon"][0]
            su = sp[:2] / np.linalg.norm(sp[:2])
            mu = mp[:2] / np.linalg.norm(mp[:2])
            # a seta aponta para FORA, na direção em que o Sol está
            sun_arrow.set_position((su[0] * 2.40, su[1] * 2.40))
            sun_arrow.xy = (su[0] * 2.78, su[1] * 2.78)
            sun_txt.set_position((su[0] * 2.95, su[1] * 2.95))
            sun_txt.set_text("Sol")
            # ângulo de fase lunar: 0 ou 180 = sizígia, 90 = quadratura
            cos_sep = float(np.clip(np.dot(su, mu), -1.0, 1.0))
            sep = np.degrees(np.arccos(cos_sep))
            aligned = min(sep, 180.0 - sep)
            phase_txt = (
                "  ·  SIZÍGIA (Lua, Terra e Sol alinhados — maré máxima)"
                if aligned < 20
                else "  ·  QUADRATURA (Sol a 90° da Lua — maré mínima)"
                if aligned > 70
                else f"  ·  separação Sol–Lua {sep:.0f}°"
            )

        title.set_text(f"Marés oceânicas — {time_label(t)}")
        subtitle.set_text(
            f"amplitude de equilíbrio {np.ptp(h):.2f} m"
            f"  ·  dinâmica {np.ptp(h) * amp:.2f} m{phase_txt}"
        )
        return (quiver, eq_line, dyn_line, meridian, moon_dot, moon_dot_f, title, subtitle)

    anim = FuncAnimation(fig, update, frames=len(idx), interval=interval, blit=False)
    return anim, fig
