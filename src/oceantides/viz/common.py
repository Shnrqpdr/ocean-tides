"""Paleta, estilo e utilidades compartilhadas pelas três animações.

**Cor.** A altura da maré é um dado de *polaridade* (acima/abaixo do nível
médio), então usa uma escala **divergente** de dois tons com cinza neutro no
meio, sempre ancorada simetricamente em zero -- nunca arco-íris, e nunca com um
tom no ponto médio. As séries das estações usam uma paleta **categórica** de
ordem fixa.

**Tema claro apenas, deliberadamente.** A paleta categórica de 4 tons
(azul, laranja, aqua, violeta) foi validada para todos os pares na superfície
clara ``#fcfcfb``: pior par com deficiência de visão de cor ΔE 9.2, pior par
com visão normal ΔE 16.3. No tema escuro nenhum quarto tom passa nos mesmos
critérios (o violeta escuro fica a ΔE 1.9 do azul escuro), então em vez de
publicar um modo escuro que falha na acessibilidade, fixamos a superfície
clara -- que, ao contrário de uma página web, é conhecida de antemão numa
figura matplotlib.

O aqua fica em 2.74:1 contra a superfície, abaixo de 3:1, o que obriga a
*relevo*: todas as linhas de série levam rótulo direto além da legenda, de
modo que a identidade nunca depende só da cor.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["PALETTE", "SERIES", "tide_cmap", "style_axes", "time_label", "auto_exaggeration"]


PALETTE = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "neutral": "#f0efec",
}

# Ordem categórica fixa -- nunca reciclada, nunca reordenada por classificação.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")  # azul, laranja, aqua, violeta


# Divergente azul <-> vermelho, cinza neutro no centro. Cada braço é
# monotônico em luminosidade, com o mesmo número de degraus.
_DIVERGING = [
    (0.000, "#0d366b"),
    (0.125, "#256abf"),
    (0.250, "#3987e5"),
    (0.375, "#9ec5f4"),
    (0.460, "#e3edfb"),
    (0.500, "#f0efec"),  # cinza neutro = nível médio exato
    (0.540, "#fbe7e5"),
    (0.625, "#f5b3b0"),
    (0.750, "#e34948"),
    (0.875, "#b02f2e"),
    (1.000, "#6e1817"),
]


def tide_cmap():
    """Colormap divergente para ``h``. Use sempre com ``vmin = -vmax``."""
    return LinearSegmentedColormap.from_list("mare", _DIVERGING)


def style_axes(ax, grid=True, spines=("left", "bottom")):
    """Eixos recessivos: grade fina, sem molduras supérfluas, texto em tinta."""
    ax.set_facecolor(PALETTE["surface"])
    for name, spine in ax.spines.items():
        if name in spines:
            spine.set_color(PALETTE["axis"])
            spine.set_linewidth(1.0)
        else:
            spine.set_visible(False)
    ax.tick_params(colors=PALETTE["ink_muted"], labelsize=9, length=3, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(PALETTE["ink_secondary"])
    if grid:
        ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=1.0)
        ax.set_axisbelow(True)
    return ax


def time_label(seconds: float) -> str:
    """'dia 12, 06:30' -- mais legível que segundos ou dias decimais."""
    total_h = seconds / 3600.0
    day = int(total_h // 24)
    rem = total_h - 24 * day
    return f"dia {day}, {int(rem):02d}:{int(round((rem % 1) * 60)) % 60:02d}"


def ptbr(value: float, decimals: int = 0) -> str:
    """Número com separador de milhar em português (1.234.567)."""
    return f"{value:,.{decimals}f}".replace(",", " ")


def auto_exaggeration(h_max: float, radius: float, target_fraction: float = 0.22):
    """Fator que torna o bojo visível, e o texto que o declara.

    A maré real vale ``h/R ~ 8e-8`` do raio da Terra: sem exagero o bojo é
    invisível. Todo gráfico que aplica o exagero **precisa** exibir o fator,
    sob pena de sugerir que os oceanos formam um elipsoide grosseiro.
    """
    if h_max <= 0:
        return 1.0, "sem exagero"
    factor = target_fraction * radius / h_max
    return factor, (
        f"bojo exagerado {ptbr(factor)}×  (real: {h_max / radius:.1e} do raio da Terra)"
    )


def phase_lag_to_angle(lag_rad: float) -> float:
    """Atraso de fase temporal -> atraso *espacial* do bojo [rad].

    O padrão de maré é de grau 2 (dois bojos por volta), então ele varre 2
    ciclos de fase a cada rotação relativa. Um atraso de fase ``delta`` na
    componente M2 corresponde a um deslocamento angular de ``delta/2``.

    É essa rotação que produz a Figura 5-13 do PDF: a maré alta não fica sobre
    o eixo Terra-Lua, mas alguns graus adiante, arrastada pela rotação.
    """
    return lag_rad / 2.0


def frame_indices(n_samples: int, frames: int | None):
    """Índices igualmente espaçados, sem nunca pular o último quadro."""
    if frames is None or frames >= n_samples:
        return np.arange(n_samples)
    return np.unique(np.linspace(0, n_samples - 1, frames).astype(int))
