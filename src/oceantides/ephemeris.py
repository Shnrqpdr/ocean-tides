"""Posições da Lua e do Sol vistas do centro da Terra.

Referencial: **equatorial geocêntrico** -- ``z`` ao longo do eixo de rotação da
Terra, ``x`` na direção do equinócio vernal. É o referencial em que a rotação
da Terra é uma simples rotação em torno de ``z``, o que barateia o cálculo da
grade (ver :mod:`oceantides.grid`).

As órbitas são construídas no plano da eclíptica e depois inclinadas pela
obliquidade. Isso é o que produz a **declinação** variável dos corpos, e daí a
*desigualdade diurna* -- o PDF menciona na p. 204 que "the plane of the moon's
orbit about Earth is also not perpendicular to Earth's rotation axis. This
causes one high tide each day to be slightly higher than the other".

Quatro modos, do mais simples ao mais fiel:

``circular``
    Hipótese do próprio PDF: órbita circular, ``D`` constante, sem inclinação.
``kepler``
    Elipse kepleriana resolvida analiticamente. ``D`` varia (perigeu/apogeu),
    e como a maré vai com ``D^-3`` isso produz as marés perigeanas.
``nbody``
    Mesma órbita, mas integrada numericamente com um dos drivers de
    :mod:`oceantides.integrators`. É o que o modo ``bench`` compara contra o
    ``kepler`` analítico.
``astropy``
    Efemérides reais (opcional, requer ``astropy``). Verdade de referência.

No instante ``t = 0`` a Lua e o Sol estão ambos sobre ``+x``: lua nova, e
portanto **maré de sizígia**. Isso torna o ciclo sizígia/quadratura imediato de
ler na animação.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    E_EARTH_ORBIT,
    E_MOON,
    GM_EARTH,
    GM_MOON,
    GM_SUN,
    I_MOON,
    OBLIQUITY,
    T_MOON_SIDEREAL,
    T_YEAR,
)
from .integrators import ORBIT_DRIVERS

__all__ = [
    "solve_kepler",
    "KeplerEphemeris",
    "NBodyEphemeris",
    "moon_ephemeris",
    "sun_ephemeris",
    "make_ephemeris",
]


# ---------------------------------------------------------------------------
# Equação de Kepler
# ---------------------------------------------------------------------------


def solve_kepler(M, e, tol=1e-13, maxiter=60):
    """Resolve ``M = E - e sin(E)`` por Newton-Raphson, vetorizado em ``M``.

    O chute inicial ``E = M + e sin(M)`` já é bom o bastante para convergir em
    ~4 iterações nas excentricidades deste projeto (0.017 e 0.055).
    """
    M = np.asarray(M, dtype=float)
    M = np.mod(M + np.pi, 2.0 * np.pi) - np.pi  # para [-pi, pi], onde converge melhor
    E = M + e * np.sin(M)

    for _ in range(maxiter):
        delta = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - delta
        if np.max(np.abs(delta)) < tol:
            break
    else:
        raise RuntimeError("equação de Kepler não convergiu")
    return E


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Efeméride kepleriana
# ---------------------------------------------------------------------------


@dataclass
class KeplerEphemeris:
    """Órbita kepleriana fechada, sem integração numérica.

    Precisão de máquina em qualquer instante, custo O(1), e zero deriva
    secular -- por isso é o padrão de produção em vez do RK4.
    """

    mu: float
    period: float
    eccentricity: float = 0.0
    inclination: float = 0.0
    raan: float = 0.0
    arg_periapsis: float = 0.0
    mean_anomaly0: float = 0.0
    obliquity: float = OBLIQUITY
    name: str = "corpo"

    _rotation: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # O semieixo vem do período (3ª lei), e não o contrário: assim os
        # períodos de maré derivados (M2 = 12h25.2min) saem exatos.
        self.semi_major = (self.mu * (self.period / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
        # eclíptica -> equatorial, depois orientação da órbita na eclíptica
        self._rotation = (
            _rot_x(self.obliquity)
            @ _rot_z(self.raan)
            @ _rot_x(self.inclination)
            @ _rot_z(self.arg_periapsis)
        )

    @property
    def mean_motion(self) -> float:
        return 2.0 * np.pi / self.period

    def position(self, t) -> np.ndarray:
        """Posição geocêntrica [m]. ``t`` escalar -> (3,); vetor -> (N, 3)."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        e = self.eccentricity

        E = solve_kepler(self.mean_anomaly0 + self.mean_motion * t_arr, e)
        # Coordenadas no plano orbital a partir da anomalia excêntrica.
        x = self.semi_major * (np.cos(E) - e)
        y = self.semi_major * np.sqrt(1.0 - e * e) * np.sin(E)
        perifocal = np.stack([x, y, np.zeros_like(x)], axis=-1)

        out = perifocal @ self._rotation.T
        return out[0] if np.ndim(t) == 0 else out

    def velocity(self, t) -> np.ndarray:
        """Velocidade geocêntrica [m/s], por diferenciação analítica."""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        e = self.eccentricity
        E = solve_kepler(self.mean_anomaly0 + self.mean_motion * t_arr, e)
        # dE/dt = n / (1 - e cos E)
        dE = self.mean_motion / (1.0 - e * np.cos(E))
        vx = -self.semi_major * np.sin(E) * dE
        vy = self.semi_major * np.sqrt(1.0 - e * e) * np.cos(E) * dE
        perifocal = np.stack([vx, vy, np.zeros_like(vx)], axis=-1)
        out = perifocal @ self._rotation.T
        return out[0] if np.ndim(t) == 0 else out

    def distance(self, t) -> np.ndarray:
        return np.linalg.norm(self.position(t), axis=-1)


# ---------------------------------------------------------------------------
# Efeméride integrada numericamente
# ---------------------------------------------------------------------------


class NBodyEphemeris:
    """Mesma órbita, obtida integrando ``r'' = -mu r/|r|^3``.

    Existe para que o modo ``bench`` possa medir a deriva de cada integrador
    contra a solução kepleriana exata. Interpola linearmente entre as amostras
    quando consultada em instantes intermediários.
    """

    def __init__(self, kepler: KeplerEphemeris, method="rk4", dt=600.0, duration=None):
        if method not in ORBIT_DRIVERS:
            raise ValueError(f"método desconhecido: {method!r}; use {list(ORBIT_DRIVERS)}")
        self.kepler = kepler
        self.method = method
        self.dt = float(dt)
        self.mu = kepler.mu
        self.name = kepler.name

        duration = float(duration if duration is not None else 2.0 * kepler.period)
        n_steps = max(1, int(round(duration / self.dt)))

        def accel(_t, x):
            r = np.linalg.norm(x)
            return -self.mu * x / r**3

        driver = ORBIT_DRIVERS[method]
        self.t, self.x, self.v = driver(
            accel, kepler.position(0.0), kepler.velocity(0.0), 0.0, self.dt, n_steps
        )

    def position(self, t) -> np.ndarray:
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.stack(
            [np.interp(t_arr, self.t, self.x[:, i]) for i in range(3)], axis=-1
        )
        return out[0] if np.ndim(t) == 0 else out

    def distance(self, t) -> np.ndarray:
        return np.linalg.norm(self.position(t), axis=-1)

    def specific_energy(self) -> np.ndarray:
        """Energia específica ``v^2/2 - mu/r`` em cada amostra; deve ser constante."""
        speed_sq = np.sum(self.v**2, axis=-1)
        radius = np.linalg.norm(self.x, axis=-1)
        return 0.5 * speed_sq - self.mu / radius


# ---------------------------------------------------------------------------
# Construtores
# ---------------------------------------------------------------------------


def moon_ephemeris(mode="kepler", **kw) -> KeplerEphemeris:
    """Lua. ``mode='circular'`` reproduz a hipótese simplificada do PDF."""
    circular = mode == "circular"
    return KeplerEphemeris(
        mu=GM_EARTH + GM_MOON,
        period=T_MOON_SIDEREAL,
        eccentricity=0.0 if circular else E_MOON,
        inclination=0.0 if circular else I_MOON,
        obliquity=0.0 if circular else OBLIQUITY,
        name="Lua",
        **kw,
    )


def sun_ephemeris(mode="kepler", **kw) -> KeplerEphemeris:
    """Sol visto da Terra (o vetor Terra->Sol tem o mesmo módulo do Sol->Terra,
    e a fase é escolhida para coincidir com a Lua em t=0)."""
    circular = mode == "circular"
    return KeplerEphemeris(
        mu=GM_SUN + GM_EARTH,
        period=T_YEAR,
        eccentricity=0.0 if circular else E_EARTH_ORBIT,
        inclination=0.0,  # define a eclíptica
        obliquity=0.0 if circular else OBLIQUITY,
        name="Sol",
        **kw,
    )


def make_ephemeris(body: str, mode: str = "kepler", **kw):
    """Fábrica: ``body`` em {'moon', 'sun'}, ``mode`` em
    {'circular', 'kepler', 'nbody', 'astropy'}."""
    build = {"moon": moon_ephemeris, "sun": sun_ephemeris}
    if body not in build:
        raise ValueError(f"corpo desconhecido: {body!r}")

    if mode == "astropy":
        return _astropy_ephemeris(body)
    if mode == "nbody":
        nbody_kw = {k: kw.pop(k) for k in ("method", "dt", "duration") if k in kw}
        return NBodyEphemeris(build[body]("kepler", **kw), **nbody_kw)
    return build[body](mode, **kw)


def _astropy_ephemeris(body: str):
    """Efemérides JPL reais. Só para validação cruzada; requer ``astropy``."""
    try:
        from astropy.coordinates import GCRS, get_body
        from astropy.time import Time
    except ImportError as exc:  # pragma: no cover - caminho opcional
        raise ImportError(
            "modo 'astropy' requer o extra: uv sync --extra astro"
        ) from exc

    import astropy.units as u

    epoch = Time("2026-01-01T00:00:00", scale="utc")

    class _AstropyEphemeris:
        name = body

        def position(self, t):
            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            times = epoch + t_arr * u.s
            coord = get_body(body, times).transform_to(GCRS(obstime=times))
            xyz = coord.cartesian.xyz.to(u.m).value.T
            return xyz[0] if np.ndim(t) == 0 else xyz

        def distance(self, t):
            return np.linalg.norm(self.position(t), axis=-1)

    return _AstropyEphemeris()
