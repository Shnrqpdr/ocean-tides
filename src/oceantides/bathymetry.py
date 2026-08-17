"""Batimetria global e máscara de terra, a partir do ETOPO 2022 (NOAA NCEI).

A maré de equilíbrio não precisa saber onde há água. Um modelo dinâmico precisa
de tudo: a profundidade fixa a velocidade da onda ``c = sqrt(gH)``, e as costas
são as fronteiras que refletem a onda e criam os sistemas anfidrômicos.

**Por que a profundidade é o parâmetro central.** Em oceano profundo
(H ~ 4000 m), ``c = 198 m/s``. O ponto sublunar varre o solo a ~448 m/s. A
onda de maré é fisicamente incapaz de acompanhar a Lua -- ela responde como uma
onda forçada, refletida e ressonante, e não como o bojo hidrostático que a
teoria de equilíbrio supõe. Todo o desvio entre os dois modelos nasce daí.

Fonte: ETOPO 2022 1 arc-minute Global Relief (bed elevation), NOAA NCEI.
https://www.ncei.noaa.gov/products/etopo-global-relief-model

Baixado uma única vez via OPeNDAP com subamostragem no servidor (5 arc-min,
~37 MB em vez de 933 MB) e guardado em cache local.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "ETOPO_URL",
    "fetch_etopo",
    "load_relief",
    "coarsen_to_grid",
    "Bathymetry",
]

ETOPO_URL = (
    "https://www.ngdc.noaa.gov/thredds/dodsC/global/ETOPO2022/60s/"
    "60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
)

# 5 arc-min: divisor exato de 0.25, 0.5 e 1 grau, então qualquer resolução
# do modelo sai de um agrupamento inteiro de células.
STRIDE = 5
DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "etopo_5min.npz"


def fetch_etopo(cache=DEFAULT_CACHE, stride=STRIDE, chunk_rows=360, verbose=True):
    """Baixa e guarda o relevo global subamostrado. Idempotente."""
    cache = Path(cache)
    if cache.exists():
        return cache

    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover
        raise ImportError("requer o extra: uv sync --extra data") from exc

    cache.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"baixando ETOPO 2022 (stride {stride}) de {ETOPO_URL}", flush=True)

    ds = xr.open_dataset(ETOPO_URL)
    lat = ds["lat"].values[::stride]
    lon = ds["lon"].values[::stride]

    n_rows = lat.size
    out = np.empty((n_rows, lon.size), dtype=np.float32)
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        out[start:stop] = ds["z"][
            start * stride : stop * stride : stride, ::stride
        ].values
        if verbose:
            print(f"  {stop}/{n_rows} linhas", flush=True)

    np.savez_compressed(cache, lat=lat, lon=lon, elevation=out)
    ds.close()
    if verbose:
        print(f"cache gravado em {cache} ({cache.stat().st_size / 1e6:.0f} MB)")
    return cache


def load_relief(cache=DEFAULT_CACHE):
    """Devolve ``(lat, lon, elevacao)`` do cache; baixa se não existir."""
    cache = Path(cache)
    if not cache.exists():
        fetch_etopo(cache)
    data = np.load(cache)
    return data["lat"], data["lon"], data["elevation"]


def coarsen_to_grid(elevation, factor):
    """Agrupa blocos ``factor x factor``, separando oceano de terra.

    A média é feita **apenas sobre as subcélulas oceânicas**: misturar cotas
    positivas de terra na média rebaixaria artificialmente a profundidade de
    células costeiras, que são justamente onde a maré é mais sensível.

    Returns
    -------
    depth : ndarray
        Profundidade média das subcélulas oceânicas [m, positiva para baixo].
        Vale 0 onde não há nenhuma subcélula oceânica.
    ocean_fraction : ndarray
        Fração de subcélulas com elevação negativa, em ``[0, 1]``.
    """
    n_lat, n_lon = elevation.shape
    if n_lat % factor or n_lon % factor:
        raise ValueError(f"grade {elevation.shape} não é divisível por {factor}")

    blocks = elevation.reshape(n_lat // factor, factor, n_lon // factor, factor)
    is_ocean = blocks < 0.0
    count = is_ocean.sum(axis=(1, 3))

    depth_sum = np.where(is_ocean, -blocks, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(count > 0, depth_sum / np.maximum(count, 1), 0.0)

    return depth, count / (factor * factor)


class Bathymetry:
    """Profundidade e máscara de oceano numa grade regular lat/lon.

    Parameters
    ----------
    resolution : float
        Tamanho da célula em graus. Precisa ser múltiplo de 5 arc-min
        (1/12 grau): 0.25, 0.5, 1.0 ...
    min_depth : float
        Profundidade mínima imposta às células oceânicas [m]. Células muito
        rasas disparam o limite CFL (``dt < dx/sqrt(gH)`` é *menos* restritivo
        em água rasa, mas o atrito quadrático explode) e são numericamente
        instáveis; 20 m é o valor usual em modelos barotrópicos globais.
    ocean_threshold : float
        Fração mínima de oceano para a célula contar como água.
    lat_limit : float
        Latitude além da qual tudo vira terra. Existe por razão numérica: a
        largura da célula encolhe com ``cos(phi)``, então o passo de tempo
        estável tende a zero no polo. Em 86 graus a célula ainda tem ~7.7 km
        a 1 grau de resolução, o que mantém ``dt`` praticável.
    """

    ARC5 = 1.0 / 12.0  # resolução da fonte em graus

    def __init__(self, resolution=1.0, min_depth=20.0, ocean_threshold=0.5,
                 lat_limit=86.0, cache=DEFAULT_CACHE):
        factor = resolution / self.ARC5
        if abs(factor - round(factor)) > 1e-9:
            raise ValueError(
                f"resolução {resolution} não é múltiplo de {self.ARC5:.6f} graus (5 arc-min)"
            )
        factor = int(round(factor))

        _, _, elevation = load_relief(cache)
        depth, ocean_fraction = coarsen_to_grid(elevation, factor)

        self.resolution = float(resolution)
        self.n_lat, self.n_lon = depth.shape
        # centros de célula
        self.lat = -90.0 + resolution * (np.arange(self.n_lat) + 0.5)
        self.lon = -180.0 + resolution * (np.arange(self.n_lon) + 0.5)

        self.ocean_fraction = ocean_fraction
        self.mask = (ocean_fraction >= ocean_threshold) & (depth > 0.0)
        self.mask &= np.abs(self.lat)[:, None] <= lat_limit

        self.depth = np.where(self.mask, np.maximum(depth, min_depth), 0.0)
        self.lat_limit = float(lat_limit)
        self.min_depth = float(min_depth)

    @property
    def shape(self):
        return (self.n_lat, self.n_lon)

    @property
    def ocean_cells(self) -> int:
        return int(self.mask.sum())

    def wave_speed(self):
        """``c = sqrt(gH)`` [m/s] -- a velocidade da onda de maré."""
        from .constants import G_SURFACE

        return np.sqrt(G_SURFACE * self.depth)

    def cell_width(self):
        """Largura zonal da célula [m], que encolhe com ``cos(latitude)``."""
        from .constants import R_EARTH

        dlon = np.radians(self.resolution)
        return R_EARTH * dlon * np.cos(np.radians(self.lat))[:, None]

    def cfl_timestep(self, safety=0.45):
        """Maior passo de tempo estável para o esquema explícito [s].

        Condição de Courant para ondas de gravidade em duas dimensões:
        ``dt <= safety * min(dx, dy) / sqrt(gH)``.
        """
        from .constants import R_EARTH

        dy = R_EARTH * np.radians(self.resolution)
        dx = self.cell_width()
        c = self.wave_speed()

        span = np.minimum(dx, dy)
        with np.errstate(invalid="ignore", divide="ignore"):
            limit = np.where(self.mask & (c > 0), span / np.maximum(c, 1e-9), np.inf)
        return float(safety * np.nanmin(limit))

    def summary(self) -> str:
        d = self.depth[self.mask]
        return (
            f"grade {self.n_lat}x{self.n_lon} a {self.resolution}°  ·  "
            f"{self.ocean_cells} células de oceano ({100 * self.mask.mean():.1f}% do globo)\n"
            f"profundidade: média {d.mean():.0f} m, mediana {np.median(d):.0f} m, "
            f"máx {d.max():.0f} m\n"
            f"velocidade de onda: {self.wave_speed()[self.mask].max():.0f} m/s máx  ·  "
            f"passo CFL: {self.cfl_timestep():.1f} s"
        )
