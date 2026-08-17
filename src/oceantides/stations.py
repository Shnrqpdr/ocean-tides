"""Marégrafos: onde a maré é observada, e com que bacia ela ressoa.

**Sobre os parâmetros ``natural_period`` e ``q_factor``.** São parâmetros
*concentrados e ajustados*, não ressonâncias medidas de bacias reais. Um único
oscilador de um grau de liberdade não consegue reproduzir simultaneamente a
geometria e a amplitude de uma baía real -- a Baía de Fundy, por exemplo, tem
ressonância observada perto de 13.3 h, mas nesse período um modelo de 1 GL não
passa de ~6.8x de amplificação, qualquer que seja o ``Q``.

O que foi ajustado é o *contraste qualitativo* que o PDF descreve na p. 203
("local effects can be dramatic... resonances can affect the natural
oscillation"): bacias quase ressonantes amplificam muito e atrasam muito;
bacias rígidas seguem o equilíbrio; bacias lentas respondem *abaixo* do
equilíbrio.

**Onde o modelo acerta e onde não acerta** (medido, ver
``tests/test_stations.py``). Nas estações de baixa latitude ele chega perto:
Rio de Janeiro 1.11 m contra 1.20 m observados, Santos 1.15 contra 1.40,
Belém 2.99 contra 3.50, e o Taiti responde corretamente *abaixo* do
equilíbrio. Já as grandes bacias ressonantes de alta latitude ficam ~3x
subestimadas -- Fundy dá 5.2 m contra os 16 m reais.

Essa discrepância não é um erro de ajuste, é física ausente do modelo. A maré
de equilíbrio *local* em 45 graus de latitude vale só ~66% da equatorial
(o ponto nunca chega perto do ponto sublunar), e os 16 m reais de Fundy vêm da
co-oscilação da Baía com o sistema anfidrômico do Atlântico Norte inteiro --
um modo de bacia oceânica, não o forçamento local. Reproduzir isso exigiria as
equações de maré de Laplace, não um oscilador de um grau de liberdade.

A coluna ``observed_range`` traz a amplitude real de sizígia apenas como
referência de comparação.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import geographic_to_ecef

__all__ = ["Station", "DEFAULT_STATIONS", "station_positions", "get_stations"]


@dataclass(frozen=True)
class Station:
    name: str
    lat: float
    lon: float
    natural_period_h: float
    q_factor: float
    observed_range: float  # amplitude real de sizígia [m], só para referência
    note: str = ""

    @property
    def natural_period(self) -> float:
        return self.natural_period_h * 3600.0

    def position(self):
        """Vetor no referencial fixo à Terra."""
        return geographic_to_ecef(self.lat, self.lon)


DEFAULT_STATIONS = (
    # --- bacias quase ressonantes: amplificação e atraso grandes -------------
    Station(
        "Burntcoat Head", 45.30, -64.00, 12.5, 20.0, 16.0,
        "Baía de Fundy: a maior maré do mundo, quase em ressonância com M2",
    ),
    Station(
        "Ungava Bay", 58.95, -69.50, 12.55, 16.0, 15.0,
        "rival de Fundy pelo título de maior maré",
    ),
    Station(
        "Avonmouth", 51.51, -2.70, 12.7, 12.0, 12.2,
        "Canal de Bristol: funil que amplifica a maré do Atlântico",
    ),
    # --- costa brasileira ----------------------------------------------------
    Station(
        "Sao Luis", -2.53, -44.30, 13.5, 8.0, 6.6,
        "maior maré do Brasil, na costa amazônica",
    ),
    Station(
        "Belem", -1.44, -48.50, 14.0, 6.0, 3.5,
        "foz do Amazonas; a maré subindo o rio produz a pororoca",
    ),
    Station(
        "Salvador", -12.97, -38.51, 8.0, 3.0, 2.3,
        "plataforma estreita, amplificação moderada",
    ),
    Station(
        "Santos", -23.96, -46.33, 7.5, 2.0, 1.4,
        "maré de micro-amplitude do Sudeste",
    ),
    Station(
        "Rio de Janeiro", -22.90, -43.17, 7.0, 2.0, 1.2,
        "bacia rígida: segue de perto a maré de equilíbrio",
    ),
    # --- casos-limite --------------------------------------------------------
    Station(
        "Papeete", -17.53, -149.57, 20.0, 1.0, 0.3,
        "Taiti: bacia lenta responde ABAIXO do equilíbrio (A < 1)",
    ),
    Station(
        "Pacifico Central", 0.0, -140.0, 4.0, 1.0, 0.5,
        "mar aberto: quase exatamente a maré de equilíbrio do PDF",
    ),
)


def get_stations(names=None):
    """Seleciona estações por nome; sem argumento devolve todas."""
    if names is None:
        return DEFAULT_STATIONS
    lookup = {s.name.lower(): s for s in DEFAULT_STATIONS}
    out = []
    for n in names:
        key = n.strip().lower()
        if key not in lookup:
            raise KeyError(f"estação desconhecida: {n!r}; conhecidas: {list(lookup)}")
        out.append(lookup[key])
    return tuple(out)


def station_positions(stations) -> np.ndarray:
    """Array ``(n, 3)`` das posições no referencial fixo à Terra."""
    return np.stack([s.position() for s in stations])
