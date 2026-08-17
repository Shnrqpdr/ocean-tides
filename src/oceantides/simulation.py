"""Orquestração: configuração -> resultado.

O que é armazenado e o que é recalculado:

* **Armazenado**: séries temporais das estações (maré de equilíbrio e maré
  dinâmica) e posições da Lua e do Sol nos instantes amostrados.
* **Recalculado sob demanda**: o campo global ``h(lat, lon)``. É forma fechada
  e custa ~2 ms por quadro; guardá-lo para 4000 quadros numa grade 181x361
  daria ~1 GB sem nenhum ganho.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .constants import GM_MOON, GM_SUN, T_M2
from .ephemeris import make_ephemeris
from .forcing import equilibrium_height
from .grid import eci_to_ecef
from .response import BasinResponse
from .stations import get_stations, station_positions

__all__ = ["SimConfig", "SimResult", "run_simulation", "build_bodies"]


@dataclass
class SimConfig:
    days: float = 30.0
    dt: float = 60.0
    sample_dt: float = 600.0
    ephemeris_mode: str = "kepler"
    bodies: tuple = ("moon", "sun")
    stations: tuple | None = None
    love: bool = False
    theta0: float = 0.0
    grid_shape: tuple = (91, 181)
    spin_up: bool = True

    def __post_init__(self):
        self.bodies = tuple(self.bodies)
        if self.stations is not None:
            self.stations = tuple(self.stations)
        self.grid_shape = tuple(self.grid_shape)

    @property
    def duration(self) -> float:
        return self.days * 86400.0

    @property
    def n_steps(self) -> int:
        return int(round(self.duration / self.dt))

    @property
    def sample_every(self) -> int:
        return max(1, int(round(self.sample_dt / self.dt)))


@dataclass
class SimResult:
    t: np.ndarray  # (n_t,) segundos
    h_dynamic: np.ndarray  # (n_t, n_stations) resposta da bacia [m]
    h_equilibrium: np.ndarray  # (n_t, n_stations) maré de equilíbrio [m]
    moon_pos: np.ndarray  # (n_t, 3) geocêntrico inercial [m]
    sun_pos: np.ndarray  # (n_t, 3)
    station_names: tuple
    config: SimConfig
    meta: dict = field(default_factory=dict)

    @property
    def days(self) -> np.ndarray:
        return self.t / 86400.0

    def station(self, name: str):
        """``(h_dinamica, h_equilibrio)`` de uma estação, pelo nome."""
        idx = [n.lower() for n in self.station_names].index(name.lower())
        return self.h_dynamic[:, idx], self.h_equilibrium[:, idx]

    def ranges(self) -> dict:
        """Amplitude pico-a-pico de cada estação [m]."""
        return {
            n: float(self.h_dynamic[:, i].max() - self.h_dynamic[:, i].min())
            for i, n in enumerate(self.station_names)
        }

    def rebuild_bodies(self):
        """Reconstrói as efemérides para recalcular o campo global na animação."""
        return build_bodies(self.config)


def build_bodies(config: SimConfig):
    """Lista de pares ``(efeméride, GM)`` conforme a configuração."""
    gm = {"moon": GM_MOON, "sun": GM_SUN}
    return [
        (make_ephemeris(b, config.ephemeris_mode), gm[b])
        for b in config.bodies
    ]


def make_forcing(points, bodies, theta0=0.0, love=False):
    """Devolve ``h_eq(t)`` como **função**, avaliável em qualquer instante.

    É deliberadamente uma função e não um array pré-amostrado: o RK4 avalia o
    forçamento em ``t + dt/2``, e amostrá-lo só nos instantes ``t`` reduziria o
    método de 4ª para 2ª ordem.
    """
    points = np.asarray(points, dtype=float)

    def h_eq(t):
        total = np.zeros(points.shape[:-1])
        for eph, gm in bodies:
            pos = eci_to_ecef(eph.position(t), t, theta0)
            total = total + equilibrium_height(points, pos, gm, love=love)
        return total

    return h_eq


def run_simulation(config: SimConfig | None = None, **kw) -> SimResult:
    """Executa a simulação completa."""
    config = config or SimConfig(**kw)

    stations = get_stations(config.stations)
    points = station_positions(stations)
    bodies = build_bodies(config)

    forcing = make_forcing(points, bodies, config.theta0, config.love)

    basin = BasinResponse(
        natural_period=[s.natural_period for s in stations],
        q_factor=[s.q_factor for s in stations],
    )

    # Spin-up: integra ANTES de t=0 e descarta, para que a saída comece em
    # regime permanente. Sem isso o oscilador aparece "subindo" ao regime nas
    # primeiras semanas, o que mascara o envelope de sizígia/quadratura.
    state0 = None
    if config.spin_up:
        t_spin = basin.spin_up_duration()
        n_spin = max(1, int(round(t_spin / config.dt)))
        _, h_s, v_s = basin.integrate(
            forcing, -n_spin * config.dt, config.dt, n_spin, sample_every=n_spin
        )
        state0 = np.stack([h_s[-1], v_s[-1]])

    t, h_dyn, _ = basin.integrate(
        forcing, 0.0, config.dt, config.n_steps,
        sample_every=config.sample_every, state0=state0,
    )

    # Maré de equilíbrio nos mesmos instantes, para comparação direta.
    h_eq = np.stack([forcing(ti) for ti in t])

    moon = next((e for e, _ in bodies if e.name == "Lua"), None)
    sun = next((e for e, _ in bodies if e.name == "Sol"), None)
    moon_pos = moon.position(t) if moon else np.zeros((t.size, 3))
    sun_pos = sun.position(t) if sun else np.zeros((t.size, 3))

    amp, lag = basin.response_at(T_M2)
    meta = {
        "m2_amplification": amp.tolist(),
        "m2_lag_hours": (lag / (2 * np.pi) * T_M2 / 3600.0).tolist(),
        "decay_time_days": (basin.decay_time / 86400.0).tolist(),
        "spin_up_days": basin.spin_up_duration() / 86400.0 if config.spin_up else 0.0,
    }

    return SimResult(
        t=t,
        h_dynamic=h_dyn,
        h_equilibrium=h_eq,
        moon_pos=moon_pos,
        sun_pos=sun_pos,
        station_names=tuple(s.name for s in stations),
        config=config,
        meta=meta,
    )
