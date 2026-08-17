"""Constituintes harmônicas: como a maré é realmente prevista.

Há dois caminhos para prever maré, e eles são filosoficamente distintos:

1. **Simular** a hidrodinâmica (as LTE em :mod:`oceantides.lte`).
2. **Decompor** o sinal em constituintes de frequência astronomicamente
   determinada, ajustando amplitude e fase de cada uma contra observação.

O caminho (2) é o que produz as tábuas de maré do mundo inteiro. A sacada, de
Kelvin e Darwin, é que as *frequências* saem da mecânica celeste com precisão
arbitrária, enquanto amplitude e fase, que dependem da hidrodinâmica local
impossível de calcular no séc. XIX, podem ser simplesmente **medidas**.

Toda constituinte é uma combinação inteira de seis frequências fundamentais --
os **números de Doodson** ``(n_tau, n_s, n_h, n_p, n_N, n_ps)``::

    tau  tempo lunar médio            14.4920521 graus/h
    s    longitude média da Lua        0.5490165
    h    longitude média do Sol        0.0410686
    p    perigeu lunar                 0.0046418
    N'   nó lunar (negativo)           0.0022064
    ps   perigeu solar                 0.0000020

A velocidade da constituinte é o produto escalar desses números com as
frequências acima. M2, por exemplo, é ``(2,0,0,0,0,0)`` = 2 x 14.4920521 =
28.9841042 graus/h, ou 12.4206 h de período -- exatamente o valor que o resto
do projeto obtém por FFT sem nunca codificá-lo.

Amplitudes de equilíbrio: desenvolvimento de Cartwright-Tayler-Edden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Constituent",
    "CONSTITUENTS",
    "MAJOR_EIGHT",
    "astronomical_arguments",
    "nodal_factors",
    "harmonic_fit",
    "predict",
    "HarmonicSolution",
]

# Frequências fundamentais em graus por hora média solar.
# O perigeu solar avança 1.7195 graus/século = 1.96e-6 graus/h; escrever
# 0.0000002 em vez de 0.0000020 desloca T2 em 1.7e-6 graus/h, erro pequeno mas
# suficiente para a constituinte não bater com a tabela de referência.
FUNDAMENTAL = np.array(
    [14.4920521, 0.5490165, 0.0410686, 0.0046418, 0.0022064, 0.0000020]
)

DEG_PER_HOUR_TO_RAD_PER_S = np.pi / 180.0 / 3600.0


@dataclass(frozen=True)
class Constituent:
    name: str
    doodson: tuple  # (n_tau, n_s, n_h, n_p, n_N, n_ps)
    amplitude: float  # amplitude de equilíbrio [m], Cartwright-Tayler-Edden
    description: str = ""
    phase_correction: float = 0.0  # graus somados a V0; ver PHASE_CORRECTION

    @property
    def speed(self) -> float:
        """Velocidade angular em graus por hora."""
        return float(np.dot(self.doodson, FUNDAMENTAL))

    @property
    def omega(self) -> float:
        """Velocidade angular em rad/s."""
        return self.speed * DEG_PER_HOUR_TO_RAD_PER_S

    @property
    def period(self) -> float:
        """Período em segundos."""
        return 2.0 * np.pi / self.omega

    @property
    def period_hours(self) -> float:
        return self.period / 3600.0

    @property
    def species(self) -> int:
        """0 = longo período, 1 = diurna, 2 = semidiurna, 4/6 = águas rasas."""
        return self.doodson[0]


# Correção de fase constante que entra em V0, por constituinte.
#
# Não é convenção arbitrária: vem do sinal das funções associadas de Legendre
# de cada espécie no desenvolvimento do potencial. As semidiurnas saem com
# fase zero; as diurnas carregam +-90 graus, com K1 de sinal oposto ao das
# demais; as de período longo, 180 graus, porque (3 sin^2(phi) - 1)/2 é
# NEGATIVO em latitudes baixas.
#
# Verificado contra as previsões oficiais da NOAA em
# `tests/test_harmonic.py::TestAgainstNOAA` -- sem essas correções a
# reconstrução da maré fica com correlação ~0.4 em vez de ~1.
PHASE_CORRECTION = {
    "K1": -90.0,
    "O1": 90.0, "P1": 90.0, "Q1": 90.0, "J1": 90.0, "M1": 90.0, "OO1": 90.0,
    "Mf": 180.0, "Mm": 180.0, "Ssa": 180.0, "Sa": 180.0, "MSf": 180.0,
}


def _c(name, doodson, amplitude, description=""):
    return Constituent(
        name, doodson, amplitude, description, PHASE_CORRECTION.get(name, 0.0)
    )


CONSTITUENTS = {
    c.name: c
    for c in (
        # --- semidiurnas (espécie 2): dominam na maior parte do planeta -----
        _c("M2", (2, 0, 0, 0, 0, 0), 0.242334, "principal lunar semidiurna"),
        _c("S2", (2, 2, -2, 0, 0, 0), 0.112841, "principal solar semidiurna"),
        _c("N2", (2, -1, 0, 1, 0, 0), 0.046398, "elíptica lunar maior (perigeu)"),
        _c("K2", (2, 2, 0, 0, 0, 0), 0.030704, "declinacional luni-solar"),
        _c("NU2", (2, -1, 2, -1, 0, 0), 0.008838, "evecional lunar maior"),
        _c("MU2", (2, -2, 2, 0, 0, 0), 0.007386, "variacional"),
        _c("L2", (2, 1, 0, -1, 0, 0), 0.006931, "elíptica lunar menor"),
        _c("T2", (2, 2, -3, 0, 0, 1), 0.006644, "elíptica solar maior"),
        _c("2N2", (2, -2, 0, 2, 0, 0), 0.006158, "elíptica lunar de 2a ordem"),
        # --- diurnas (espécie 1): dominam onde a maré é diurna --------------
        _c("K1", (1, 1, 0, 0, 0, 0), 0.141565, "declinacional luni-solar"),
        _c("O1", (1, -1, 0, 0, 0, 0), 0.100514, "principal lunar diurna"),
        _c("P1", (1, 1, -2, 0, 0, 0), 0.046843, "principal solar diurna"),
        _c("Q1", (1, -2, 0, 1, 0, 0), 0.019256, "elíptica lunar maior diurna"),
        _c("J1", (1, 2, 0, -1, 0, 0), 0.007965, "elíptica lunar menor diurna"),
        _c("M1", (1, 0, 0, 1, 0, 0), 0.007915, "lunar diurna menor"),
        _c("OO1", (1, 3, 0, 0, 0, 0), 0.004350, "declinacional lunar de 2a ordem"),
        # --- longo período (espécie 0) --------------------------------------
        _c("Mf", (0, 2, 0, 0, 0, 0), 0.041742, "quinzenal lunar"),
        _c("Mm", (0, 1, 0, -1, 0, 0), 0.022026, "mensal lunar"),
        _c("Ssa", (0, 0, 2, 0, 0, 0), 0.019446, "semianual solar"),
        _c("Sa", (0, 0, 1, 0, 0, 0), 0.003104, "anual solar"),
        _c("MSf", (0, 2, -2, 0, 0, 0), 0.006663, "mensal luni-solar sinódica"),
        # --- águas rasas: NÃO existem no potencial astronômico ---------------
        # Nascem da não-linearidade (atrito quadrático, continuidade) em mares
        # rasos. Amplitude de equilíbrio zero por definição: são geradas pela
        # própria hidrodinâmica, e é por isso que só um modelo dinâmico as
        # produz -- a teoria de equilíbrio jamais poderia.
        _c("M4", (4, 0, 0, 0, 0, 0), 0.0, "sobremaré de M2 (águas rasas)"),
        _c("MS4", (4, 2, -2, 0, 0, 0), 0.0, "composta M2+S2 (águas rasas)"),
        _c("MN4", (4, -1, 0, 1, 0, 0), 0.0, "composta M2+N2 (águas rasas)"),
        _c("M6", (6, 0, 0, 0, 0, 0), 0.0, "segunda sobremaré de M2"),
        _c("2SM2", (2, 4, -4, 0, 0, 0), 0.0, "composta (águas rasas)"),
    )
}

# As oito que os modelos globais sempre reportam e contra as quais a acurácia
# é medida (Stammer et al. 2014).
MAJOR_EIGHT = ("M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1")


# ---------------------------------------------------------------------------
# Argumentos astronômicos e fatores nodais
# ---------------------------------------------------------------------------


def astronomical_arguments(days_since_j2000):
    """Longitudes médias ``(s, h, p, N, ps)`` em graus.

    Polinômios em séculos julianos desde J2000.0, seguindo a convenção usual
    da análise de marés (Schureman / Doodson).
    """
    t = np.asarray(days_since_j2000, dtype=float) / 36525.0

    s = 218.3164477 + 481267.88123421 * t - 0.0015786 * t**2  # Lua
    h = 280.4664567 + 36000.76982779 * t + 0.0003032 * t**2  # Sol
    p = 83.3532465 + 4069.0137287 * t - 0.0103200 * t**2  # perigeu lunar
    n = 125.0445479 - 1934.1362891 * t + 0.0020754 * t**2  # nó ascendente
    ps = 282.9384000 + 1.7195000 * t  # perigeu solar

    return tuple(np.mod(x, 360.0) for x in (s, h, p, n, ps))


def mean_lunar_time(days_since_j2000):
    """Tempo lunar médio ``tau`` em graus.

    Relação de Doodson: ``tau = T + h - s``, com ``T = 15 * (hora UT) + 180``.
    É o ângulo horário médio da Lua em Greenwich -- a variável que dá o "2" de
    M2 e faz a maré ter duas altas por dia lunar, e não por dia solar.
    """
    d = np.asarray(days_since_j2000, dtype=float)
    ut_hours = 24.0 * np.mod(d + 0.5, 1.0)  # J2000 começa ao meio-dia
    big_t = 15.0 * ut_hours + 180.0
    s, h, _, _, _ = astronomical_arguments(d)
    return np.mod(big_t + h - s, 360.0)


def astronomical_argument(name, days_since_j2000, include_nodal=True):
    """Argumento astronômico ``V0 + u`` da constituinte, em graus.

    É o que falta para converter uma fase relativa a uma época arbitrária na
    **fase de Greenwich** ``g`` que marégrafos publicam, na convenção

        h(t) = f * H * cos(V0 + u - g)

    Nenhuma das constituintes desta tabela tem coeficiente não nulo em ``N'``
    (o quinto número de Doodson é 5 em todas), então o nó lunar entra apenas
    pelos fatores ``f`` e ``u``, que é o tratamento padrão.
    """
    con = CONSTITUENTS[name]
    d = np.asarray(days_since_j2000, dtype=float)
    tau = mean_lunar_time(d)
    s, h, p, node, ps = astronomical_arguments(d)

    v0 = (
        con.doodson[0] * tau
        + con.doodson[1] * s
        + con.doodson[2] * h
        + con.doodson[3] * p
        + con.doodson[4] * (-node)
        + con.doodson[5] * ps
        + con.phase_correction
    )
    if include_nodal:
        _, u = nodal_factors(name, d)
        v0 = v0 + u
    return np.mod(v0, 360.0)


def predict_from_constants(constants, days_since_j2000, names=None, mean=0.0):
    """Reconstrói a maré a partir de constantes harmônicas **publicadas**.

    É literalmente o algoritmo de uma tábua de maré::

        h(t) = z0 + sum_i f_i H_i cos(V0_i + u_i - g_i)

    Parameters
    ----------
    constants : dict
        ``{nome: (amplitude, fase_Greenwich_em_graus)}``, como a API da NOAA
        entrega.
    days_since_j2000 : array_like
        Instantes desejados.
    """
    d = np.asarray(days_since_j2000, dtype=float)
    total = np.full(d.shape, float(mean))
    for name, (amp, phase) in constants.items():
        if name not in CONSTITUENTS:
            continue  # a NOAA publica algumas que não estão nesta tabela
        if names is not None and name not in names:
            continue
        f, _ = nodal_factors(name, d)
        argument = astronomical_argument(name, d)
        total = total + f * amp * np.cos(np.radians(argument - phase))
    return total


def nodal_factors(name, days_since_j2000):
    """Fator de amplitude ``f`` e correção de fase ``u`` [graus].

    A órbita lunar precessa com período de **18.6 anos**, e isso modula a
    amplitude das constituintes: K1 varia +-11.5%, M2 apenas +-3.7%. Sem essa
    correção, uma análise harmônica de um ano dá constantes que não valem para
    o ano seguinte.

    Fórmulas de Schureman (1958), em função da longitude do nó ``N``.
    """
    _, _, _, n_deg, _ = astronomical_arguments(days_since_j2000)
    n = np.radians(n_deg)

    table = {
        # constituinte: (f, u em graus)
        "M2": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "N2": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "NU2": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "MU2": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "2N2": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "L2": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "S2": (np.ones_like(n), np.zeros_like(n)),
        "T2": (np.ones_like(n), np.zeros_like(n)),
        "K2": (
            1.024 + 0.286 * np.cos(n),
            -17.74 * np.sin(n) + 0.68 * np.sin(2 * n),
        ),
        "K1": (
            1.006 + 0.115 * np.cos(n) - 0.009 * np.cos(2 * n),
            -8.86 * np.sin(n) + 0.68 * np.sin(2 * n),
        ),
        "O1": (
            1.009 + 0.187 * np.cos(n) - 0.015 * np.cos(2 * n),
            10.80 * np.sin(n) - 1.34 * np.sin(2 * n),
        ),
        "Q1": (
            1.009 + 0.187 * np.cos(n) - 0.015 * np.cos(2 * n),
            10.80 * np.sin(n) - 1.34 * np.sin(2 * n),
        ),
        "J1": (1.013 + 0.168 * np.cos(n), -12.94 * np.sin(n)),
        "OO1": (1.030 + 0.630 * np.cos(n), -36.68 * np.sin(n)),
        "P1": (np.ones_like(n), np.zeros_like(n)),
        "M1": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "Mf": (1.043 + 0.414 * np.cos(n), -23.74 * np.sin(n)),
        "Mm": (1.0 - 0.130 * np.cos(n), np.zeros_like(n)),
        "MSf": (1.0 - 0.037 * np.cos(n), -2.14 * np.sin(n)),
        "Ssa": (np.ones_like(n), np.zeros_like(n)),
        "Sa": (np.ones_like(n), np.zeros_like(n)),
    }
    # Constituintes de águas rasas herdam o fator do harmônico gerador
    shallow = {"M4": ("M2", 2), "M6": ("M2", 3), "MN4": ("M2", 2),
               "MS4": ("M2", 1), "2SM2": ("M2", 1)}
    if name in shallow:
        parent, power = shallow[name]
        f, u = table[parent]
        return f**power, u * power

    if name not in table:
        return np.ones_like(n), np.zeros_like(n)
    return table[name]


# ---------------------------------------------------------------------------
# Análise e previsão
# ---------------------------------------------------------------------------


def harmonic_fit(t, series, names, include_mean=True):
    """Mínimos quadrados de ``series(t)`` sobre as constituintes dadas.

    É o método de Darwin: as frequências são impostas pela astronomia, e só
    amplitude e fase são ajustadas. Isso é o que torna a previsão possível a
    partir de um registro curto -- tipicamente 29 dias bastam para separar as
    principais semidiurnas e diurnas.

    Returns
    -------
    dict
        ``{nome: (amplitude, fase_em_graus)}``, mais ``'mean'`` se pedido.
        A fase segue a convenção ``h = A cos(sigma t - g)`` com ``t`` contado
        da mesma origem passada em ``t``.
    """
    t = np.asarray(t, dtype=float)
    series = np.asarray(series, dtype=float)

    cols, labels = [], []
    if include_mean:
        cols.append(np.ones_like(t))
        labels.append(("mean", None))
    for name in names:
        omega = CONSTITUENTS[name].omega
        cols.append(np.cos(omega * t))
        cols.append(np.sin(omega * t))
        labels += [(name, "cos"), (name, "sin")]

    design = np.stack(cols, axis=-1)
    coeffs, *_ = np.linalg.lstsq(design, series, rcond=None)

    out = {}
    idx = 0
    if include_mean:
        out["mean"] = float(coeffs[0])
        idx = 1
    for k, name in enumerate(names):
        c, s = coeffs[idx + 2 * k], coeffs[idx + 2 * k + 1]
        out[name] = (float(np.hypot(c, s)), float(np.degrees(np.arctan2(s, c)) % 360.0))
    return out


def predict(t, constants, mean=0.0):
    """Reconstrói a maré a partir de constantes harmônicas.

    É literalmente o algoritmo de uma tábua de maré: some as constituintes com
    as amplitudes e fases medidas naquele porto.
    """
    t = np.asarray(t, dtype=float)
    total = np.full(t.shape, float(mean))
    for name, (amp, phase) in constants.items():
        if name == "mean":
            continue
        omega = CONSTITUENTS[name].omega
        total = total + amp * np.cos(omega * t - np.radians(phase))
    return total


# ---------------------------------------------------------------------------
# Solução do modelo dinâmico
# ---------------------------------------------------------------------------


def _cluster(hits, merge_degrees):
    """Funde detecções vizinhas num único anfidromo.

    Um anfidromo largo dispara o critério de voltas em várias células
    adjacentes; sem fundir, um único ponto seria contado quatro ou cinco
    vezes. O representante do grupo é a célula de menor amplitude, que é a que
    melhor aproxima o zero verdadeiro.
    """
    remaining = sorted(hits, key=lambda h: h[2])  # menor amplitude primeiro
    clusters = []
    while remaining:
        seed = remaining.pop(0)
        keep = []
        for other in remaining:
            dlat = abs(other[0] - seed[0])
            dlon = abs(other[1] - seed[1])
            dlon = min(dlon, 360.0 - dlon)  # longitude é periódica
            if max(dlat, dlon) > merge_degrees:
                keep.append(other)
        remaining = keep
        clusters.append(seed)
    return clusters


@dataclass
class HarmonicSolution:
    """Campos de amplitude e fase por constituinte, saídos das LTE."""

    names: list
    cos_coeff: np.ndarray  # (n_con, n_lat, n_lon)
    sin_coeff: np.ndarray
    mask: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    depth: np.ndarray
    n_samples: int
    dt: float
    u_cos: np.ndarray | None = None
    u_sin: np.ndarray | None = None
    v_cos: np.ndarray | None = None
    v_sin: np.ndarray | None = None
    config: object = None
    snapshots: np.ndarray | None = None
    snapshot_times: np.ndarray | None = None

    def index(self, name: str) -> int:
        return self.names.index(name)

    def current_at(self, name: str, t):
        """Corrente de maré ``(u, v)`` [m/s] no instante ``t``.

        As correntes contam uma história diferente da elevação: são máximas
        onde a elevação é *mínima*. Num ponto anfidrômico a maré não sobe, mas
        a água corre -- a energia passa por ali mesmo assim.
        """
        if self.u_cos is None:
            raise ValueError("esta solução não guardou harmônicos de corrente")
        k = self.index(name)
        omega = CONSTITUENTS[name].omega
        c, s = np.cos(omega * t), np.sin(omega * t)
        u = self.u_cos[k] * c + self.u_sin[k] * s
        v = self.v_cos[k] * c + self.v_sin[k] * s
        return np.where(self.mask, u, np.nan), np.where(self.mask, v, np.nan)

    def current_amplitude(self, name: str):
        """Módulo da elipse de maré: maior velocidade atingida no ciclo [m/s]."""
        if self.u_cos is None:
            raise ValueError("esta solução não guardou harmônicos de corrente")
        k = self.index(name)
        # máximo de |(u,v)(t)| ao longo do ciclo, obtido analiticamente
        a, b = self.u_cos[k], self.u_sin[k]
        c, d = self.v_cos[k], self.v_sin[k]
        p = a * a + b * b + c * c + d * d
        q = np.hypot(a * a - b * b + c * c - d * d, 2.0 * (a * b + c * d))
        return np.where(self.mask, np.sqrt(np.maximum(0.5 * (p + q), 0.0)), np.nan)

    def constituent(self, name: str):
        """``(amplitude [m], fase [graus])`` da constituinte, mascarados."""
        k = self.index(name)
        c, s = self.cos_coeff[k], self.sin_coeff[k]
        amp = np.hypot(c, s)
        phase = np.degrees(np.arctan2(s, c)) % 360.0
        return (
            np.where(self.mask, amp, np.nan),
            np.where(self.mask, phase, np.nan),
        )

    def elevation_at(self, name: str, t):
        """Reconstrói o campo de elevação daquela constituinte no instante t."""
        k = self.index(name)
        omega = CONSTITUENTS[name].omega
        field = self.cos_coeff[k] * np.cos(omega * t) + self.sin_coeff[k] * np.sin(omega * t)
        return np.where(self.mask, field, np.nan)

    def sample(self, name: str, lat: float, lon: float):
        """Amplitude e fase no ponto de grade mais próximo."""
        j = int(np.argmin(np.abs(self.lat - lat)))
        i = int(np.argmin(np.abs(self.lon - lon)))
        amp, phase = self.constituent(name)
        return float(amp[j, i]), float(phase[j, i])

    def find_amphidromes(self, name: str, amplitude_threshold=0.05, min_depth=200.0,
                         merge_degrees=6.0):
        """Localiza pontos anfidrômicos pelo **número de voltas da fase**.

        A definição rigorosa de um anfidromo não é "amplitude pequena" -- isso
        também acontece em baías fechadas e em ruído costeiro. É que a fase
        percorre exatamente **360 graus** ao dar uma volta em torno do ponto.
        Somando as diferenças de fase (cada uma reduzida ao intervalo
        ``(-180, 180]``) ao longo dos oito vizinhos, o total vale ``+-360`` num
        anfidromo verdadeiro e ``0`` em qualquer outro lugar.

        O sinal informa o **sentido de rotação** da onda ao redor do ponto,
        que a força de Coriolis torna majoritariamente anti-horário no
        hemisfério norte.

        Returns
        -------
        list de ``(lat, lon, amplitude, sentido)``, com ``sentido`` em
        ``{+1, -1}`` para anti-horário e horário.
        """
        amp, phase = self.constituent(name)
        ocean = self.mask & (self.depth >= min_depth)

        # anel de oito vizinhos, em ordem ao redor da célula
        ring = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        shifted, valid = [], np.ones_like(ocean)
        for dj, di in ring:
            p = np.roll(np.roll(phase, -dj, axis=0), -di, axis=1)
            o = np.roll(np.roll(ocean, -dj, axis=0), -di, axis=1)
            if dj == -1:
                p[-1] = np.nan
                o[-1] = False
            elif dj == 1:
                p[0] = np.nan
                o[0] = False
            shifted.append(p)
            valid &= o

        winding = np.zeros_like(phase)
        for k in range(len(ring)):
            diff = shifted[(k + 1) % len(ring)] - shifted[k]
            winding += ((diff + 180.0) % 360.0) - 180.0

        with np.errstate(invalid="ignore"):
            is_amph = (
                valid
                & ocean
                & np.isfinite(winding)
                & (np.abs(np.abs(winding) - 360.0) < 90.0)
                & (amp <= amplitude_threshold)
            )
        is_amph[0] = is_amph[-1] = False

        js, iss = np.nonzero(is_amph)
        hits = [
            (float(self.lat[j]), float(self.lon[i]), float(amp[j, i]),
             int(np.sign(winding[j, i])))
            for j, i in zip(js, iss)
        ]
        return _cluster(hits, merge_degrees)

    def amphidrome_rotation_stats(self, name: str, **kw) -> dict:
        """Sentido de rotação dos anfidromos, por hemisfério.

        Serve de teste físico: sob a força de Coriolis, a onda de Kelvin
        costeira circula com a costa à direita no hemisfério norte, o que faz
        os anfidromos girarem majoritariamente no sentido **anti-horário** ao
        norte e **horário** ao sul.
        """
        found = self.find_amphidromes(name, **kw)
        out = {}
        for label, sel in (
            ("norte", [a for a in found if a[0] > 5]),
            ("sul", [a for a in found if a[0] < -5]),
        ):
            ccw = sum(1 for a in sel if a[3] > 0)
            out[label] = {
                "total": len(sel),
                "anti_horario": ccw,
                "horario": len(sel) - ccw,
                "fracao_esperada": (ccw / len(sel)) if sel and label == "norte"
                else ((len(sel) - ccw) / len(sel)) if sel else float("nan"),
            }
        return out

    def global_stats(self, name: str) -> dict:
        amp, _ = self.constituent(name)
        deep = self.mask & (self.depth > 1000.0)
        return {
            "amplitude_media": float(np.nanmean(amp)),
            "amplitude_maxima": float(np.nanmax(amp)),
            "amplitude_media_oceano_profundo": float(np.nanmean(np.where(deep, amp, np.nan))),
            "celulas": int(self.mask.sum()),
        }
