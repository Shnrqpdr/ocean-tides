"""Persistência dos resultados em ``.npz`` (opcionalmente netCDF)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .simulation import SimConfig, SimResult

__all__ = [
    "save_result",
    "load_result",
    "to_xarray",
    "save_solution",
    "load_solution",
]


def save_result(result: SimResult, path) -> Path:
    """Grava em ``.npz``. A configuração vai como JSON para permitir
    reconstruir as efemérides e recalcular o campo global na animação."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=result.t,
        h_dynamic=result.h_dynamic,
        h_equilibrium=result.h_equilibrium,
        moon_pos=result.moon_pos,
        sun_pos=result.sun_pos,
        station_names=np.array(result.station_names, dtype=object),
        config_json=json.dumps(asdict(result.config)),
        meta_json=json.dumps(result.meta),
    )
    return path


def load_result(path) -> SimResult:
    data = np.load(Path(path), allow_pickle=True)
    cfg = json.loads(str(data["config_json"]))
    for key in ("bodies", "grid_shape"):
        cfg[key] = tuple(cfg[key])
    if cfg.get("stations") is not None:
        cfg["stations"] = tuple(cfg["stations"])

    return SimResult(
        t=data["t"],
        h_dynamic=data["h_dynamic"],
        h_equilibrium=data["h_equilibrium"],
        moon_pos=data["moon_pos"],
        sun_pos=data["sun_pos"],
        station_names=tuple(data["station_names"].tolist()),
        config=SimConfig(**cfg),
        meta=json.loads(str(data["meta_json"])),
    )


def save_solution(solution, path) -> Path:
    """Grava uma :class:`~oceantides.harmonic.HarmonicSolution` das LTE.

    Só os coeficientes harmônicos vão para o disco. O campo instantâneo em
    qualquer ``t`` é reconstruído deles em uma linha -- guardar a série
    temporal completa seria centenas de vezes maior e não acrescentaria nada.
    """
    from dataclasses import asdict as _asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "cos_coeff": solution.cos_coeff,
        "sin_coeff": solution.sin_coeff,
        "mask": solution.mask,
        "lat": solution.lat,
        "lon": solution.lon,
        "depth": solution.depth,
    }
    for name in ("u_cos", "u_sin", "v_cos", "v_sin"):
        value = getattr(solution, name, None)
        if value is not None:
            arrays[name] = value

    np.savez_compressed(
        path,
        names=np.array(solution.names, dtype=object),
        n_samples=solution.n_samples,
        dt=solution.dt,
        config_json=json.dumps(
            _asdict(solution.config) if solution.config is not None else {}
        ),
        **arrays,
    )
    return path


def load_solution(path):
    from .harmonic import HarmonicSolution
    from .lte import LTEConfig

    data = np.load(Path(path), allow_pickle=True)
    cfg_raw = json.loads(str(data["config_json"]))
    config = None
    if cfg_raw:
        cfg_raw["bodies"] = tuple(cfg_raw.get("bodies", ()))
        config = LTEConfig(**cfg_raw)

    optional = {
        name: data[name] for name in ("u_cos", "u_sin", "v_cos", "v_sin")
        if name in data.files
    }
    return HarmonicSolution(
        names=list(data["names"].tolist()),
        cos_coeff=data["cos_coeff"],
        sin_coeff=data["sin_coeff"],
        mask=data["mask"],
        lat=data["lat"],
        lon=data["lon"],
        depth=data["depth"],
        n_samples=int(data["n_samples"]),
        dt=float(data["dt"]),
        config=config,
        **optional,
    )


def to_xarray(result: SimResult):
    """Converte para ``xarray.Dataset`` (requer o extra ``data``)."""
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - caminho opcional
        raise ImportError("requer o extra: uv sync --extra data") from exc

    coords = {"time": result.t / 3600.0, "station": list(result.station_names)}
    return xr.Dataset(
        {
            "h_dynamic": (("time", "station"), result.h_dynamic),
            "h_equilibrium": (("time", "station"), result.h_equilibrium),
            "moon_pos": (("time", "xyz"), result.moon_pos),
            "sun_pos": (("time", "xyz"), result.sun_pos),
        },
        coords={**coords, "xyz": ["x", "y", "z"]},
        attrs={"units_time": "hours", "units_h": "meters", **result.meta},
    )
