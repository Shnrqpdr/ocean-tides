"""Validação contra constantes harmônicas observadas (NOAA CO-OPS).

Um modelo de maré sem validação contra observação é uma opinião. Este módulo
busca as **constantes harmônicas publicadas** -- amplitude e fase de cada
constituinte, obtidas de anos de marégrafo -- e as compara com o que as LTE
produzem.

**Por que estações insulares.** Num modelo global de 1 grau, uma célula tem
~110 km: portos, estuários e plataformas estreitas simplesmente não existem na
grade. Ilhas oceânicas isoladas, ao contrário, amostram a maré de oceano
aberto, que é justamente o que um modelo barotrópico grosseiro tem chance de
acertar. Comparar contra Anchorage seria comparar contra física que o modelo
não contém; contra Midway, é uma comparação honesta.

API: https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{id}/harcon.json
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "RefStation",
    "OCEAN_STATIONS",
    "COASTAL_STATIONS",
    "fetch_constants",
    "fetch_predictions",
    "load_reference",
    "compare",
    "format_comparison",
]

HARCON_URL = (
    "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/"
    "{station}/harcon.json?units=metric"
)

DEFAULT_CACHE = (
    Path(__file__).resolve().parent.parent.parent / "data" / "noaa_harcon.json"
)


@dataclass(frozen=True)
class RefStation:
    station_id: str
    name: str
    lat: float
    lon: float
    kind: str  # "ilha" ou "costeira"


# Ilhas em oceano aberto: o alvo justo para um modelo global grosseiro.
OCEAN_STATIONS = (
    RefStation("1612340", "Honolulu, HI", 21.3067, -157.8670, "ilha"),
    RefStation("1619910", "Sand Island, Midway", 28.2117, -177.3600, "ilha"),
    RefStation("1770000", "Pago Pago, Samoa", -14.2767, -170.6900, "ilha"),
    RefStation("1630000", "Apra Harbor, Guam", 13.4433, 144.6564, "ilha"),
    RefStation("1611400", "Nawiliwili, Kauai", 21.9544, -159.3561, "ilha"),
    RefStation("1617760", "Hilo, Hawaii", 19.7303, -155.0553, "ilha"),
    RefStation("1820000", "Kwajalein", 8.7317, 167.7361, "ilha"),
    RefStation("1890000", "Wake Island", 19.2906, 166.6175, "ilha"),
)

# Estações costeiras: incluídas deliberadamente para *mostrar* onde o modelo
# global falha, não para fingir que acerta.
COASTAL_STATIONS = (
    RefStation("9414290", "San Francisco, CA", 37.8063, -122.4659, "costeira"),
    RefStation("8443970", "Boston, MA", 42.3548, -71.0534, "costeira"),
    RefStation("8724580", "Key West, FL", 24.5551, -81.8079, "costeira"),
    RefStation("9455920", "Anchorage, AK", 61.2381, -149.8892, "costeira"),
    RefStation("8518750", "The Battery, NY", 40.7006, -74.0142, "costeira"),
)

ALL_STATIONS = OCEAN_STATIONS + COASTAL_STATIONS


def fetch_constants(station_id: str, timeout=60) -> dict:
    """Busca as constantes harmônicas de uma estação. ``{nome: (amp, fase)}``."""
    with urllib.request.urlopen(HARCON_URL.format(station=station_id), timeout=timeout) as fh:
        payload = json.load(fh)
    return {
        c["name"]: (float(c["amplitude"]), float(c["phase_GMT"]))
        for c in payload.get("HarmonicConstituents", [])
    }


PREDICTION_URL = (
    "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=predictions"
    "&application=oceantides&begin_date={begin}&end_date={end}&datum=MSL"
    "&station={station}&time_zone=gmt&units=metric&interval=h&format=json"
)

PREDICTION_CACHE = (
    Path(__file__).resolve().parent.parent.parent / "data" / "noaa_predictions.json"
)

J2000 = "2000-01-01T12:00:00"


def fetch_predictions(station_id, begin="20260301", end="20260331",
                      cache=PREDICTION_CACHE, timeout=120):
    """Previsões horárias oficiais da NOAA, em dias desde J2000 e metros.

    Servem de **verdade independente** para a máquina harmônica: se
    :func:`~oceantides.harmonic.predict_from_constants` reproduz estas séries a
    partir das constantes publicadas, então argumentos astronômicos, fatores
    nodais e correções de fase estão todos certos.
    """
    import datetime as _dt

    cache = Path(cache)
    key = f"{station_id}:{begin}:{end}"
    store = json.loads(cache.read_text()) if cache.exists() else {}

    if key not in store:
        url = PREDICTION_URL.format(station=station_id, begin=begin, end=end)
        with urllib.request.urlopen(url, timeout=timeout) as fh:
            payload = json.load(fh)
        epoch = _dt.datetime(2000, 1, 1, 12)
        rows = [
            (
                (_dt.datetime.strptime(p["t"], "%Y-%m-%d %H:%M") - epoch).total_seconds()
                / 86400.0,
                float(p["v"]),
            )
            for p in payload["predictions"]
        ]
        store[key] = rows
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(store))

    rows = store[key]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def load_reference(cache=DEFAULT_CACHE, stations=ALL_STATIONS, refresh=False, verbose=True):
    """Carrega (baixando na primeira vez) as constantes de todas as estações."""
    cache = Path(cache)
    data = {}
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())

    missing = [s for s in stations if s.station_id not in data]
    if missing:
        for s in missing:
            try:
                data[s.station_id] = fetch_constants(s.station_id)
                if verbose:
                    m2 = data[s.station_id].get("M2", (float("nan"),))[0]
                    print(f"  {s.name:<26} M2 = {m2:.3f} m", flush=True)
            except Exception as exc:  # rede indisponível não deve quebrar o pipeline
                if verbose:
                    print(f"  {s.name:<26} FALHOU: {exc}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1, sort_keys=True))

    return data


def compare(solution, constituent="M2", stations=ALL_STATIONS, cache=DEFAULT_CACHE,
            search_radius=3):
    """Compara amplitude modelada e observada em cada estação.

    A amostragem busca a célula de **oceano** mais próxima dentro de
    ``search_radius`` células: numa grade de 1 grau, o ponto exato de uma
    estação costeira quase sempre cai em terra.
    """
    reference = load_reference(cache, stations, verbose=False)
    amp_field, phase_field = solution.constituent(constituent)

    rows = []
    for st in stations:
        obs = reference.get(st.station_id, {}).get(constituent)
        if obs is None:
            continue

        j = int(np.argmin(np.abs(solution.lat - st.lat)))
        i = int(np.argmin(np.abs(solution.lon - st.lon)))

        model_amp = _nearest_ocean(amp_field, j, i, search_radius)
        if model_amp is None:
            rows.append((st, obs[0], np.nan, np.nan))
            continue

        rows.append((st, obs[0], model_amp, model_amp - obs[0]))
    return rows


def _nearest_ocean(field, j, i, radius):
    """Valor da célula de oceano mais próxima, em anéis crescentes."""
    n_lat, n_lon = field.shape
    for r in range(radius + 1):
        best, best_d = None, None
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if max(abs(dj), abs(di)) != r:
                    continue
                jj, ii = j + dj, (i + di) % n_lon
                if not (0 <= jj < n_lat):
                    continue
                v = field[jj, ii]
                if np.isfinite(v):
                    d = dj * dj + di * di
                    if best_d is None or d < best_d:
                        best, best_d = float(v), d
        if best is not None:
            return best
    return None


def format_comparison(rows, constituent="M2") -> str:
    lines = [
        f"Amplitude de {constituent}: modelo vs. marégrafo (NOAA CO-OPS)",
        "",
        f"{'estacao':<26}{'tipo':<10}{'obs[m]':>9}{'modelo[m]':>11}"
        f"{'erro[m]':>10}{'erro[%]':>9}",
        "-" * 75,
    ]
    for st, obs, model, err in rows:
        if not np.isfinite(model):
            lines.append(f"{st.name:<26}{st.kind:<10}{obs:>9.3f}{'sem agua':>11}")
            continue
        lines.append(
            f"{st.name:<26}{st.kind:<10}{obs:>9.3f}{model:>11.3f}"
            f"{err:>10.3f}{100 * err / obs:>9.1f}"
        )

    for kind in ("ilha", "costeira"):
        sel = [(o, m) for st, o, m, _ in rows if st.kind == kind and np.isfinite(m)]
        if not sel:
            continue
        obs = np.array([o for o, _ in sel])
        mod = np.array([m for _, m in sel])
        rms = float(np.sqrt(np.mean((mod - obs) ** 2)))
        lines.append(
            f"\n{kind:>10}: RMS {100 * rms:.1f} cm sobre {len(sel)} estacoes  "
            f"(obs media {100 * obs.mean():.1f} cm)"
        )
    return "\n".join(lines)
