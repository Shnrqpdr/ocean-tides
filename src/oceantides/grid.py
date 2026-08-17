"""Grade geográfica e rotação da Terra.

**Otimização central deste módulo.** A maré de equilíbrio depende só do ângulo
``psi`` entre cada ponto da superfície e o eixo Terra-corpo. Poderíamos girar os
~65 mil pontos da grade para o referencial inercial a cada quadro; em vez disso
giramos **a Lua e o Sol** para o referencial fixo à Terra -- um vetor em vez de
65 mil. A grade em coordenadas terrestres é calculada uma única vez.

Ambos os referenciais compartilham o eixo ``z`` (o eixo de rotação), então a
transformação é uma rotação simples em torno de ``z`` pelo ângulo sideral de
Greenwich ``theta_g(t) = theta_0 + omega_E t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import OMEGA_EARTH, R_EARTH
from .forcing import equilibrium_height

__all__ = ["Grid", "eci_to_ecef", "greenwich_angle", "subbody_point", "geographic_to_ecef"]


def greenwich_angle(t, theta0: float = 0.0) -> np.ndarray:
    """Ângulo sideral de Greenwich [rad] no instante ``t`` [s]."""
    return theta0 + OMEGA_EARTH * np.asarray(t, dtype=float)


def eci_to_ecef(vec, t, theta0: float = 0.0) -> np.ndarray:
    """Leva um vetor do referencial inercial para o fixo à Terra.

    É uma rotação de ``-theta_g`` em torno de ``z``. Aplicada aos corpos
    perturbadores, e não à grade -- ver a nota no topo do módulo.
    """
    vec = np.asarray(vec, dtype=float)
    ang = greenwich_angle(t, theta0)
    c, s = np.cos(ang), np.sin(ang)
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([c * x + s * y, -s * x + c * y, z], axis=-1)


def geographic_to_ecef(lat_deg, lon_deg, radius=R_EARTH) -> np.ndarray:
    """(latitude, longitude) em graus -> vetor cartesiano fixo à Terra."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    cos_lat = np.cos(lat)
    return radius * np.stack(
        [cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=-1
    )


def subbody_point(body_pos_eci, t, theta0: float = 0.0):
    """Latitude e longitude do ponto sob o corpo (ponto sublunar/subsolar).

    É o centro do bojo de maré. A latitude oscila com a declinação do corpo --
    a origem da desigualdade diurna citada na p. 204 do PDF.
    """
    ecef = eci_to_ecef(body_pos_eci, t, theta0)
    r = np.linalg.norm(ecef, axis=-1)
    lat = np.degrees(np.arcsin(ecef[..., 2] / r))
    lon = np.degrees(np.arctan2(ecef[..., 1], ecef[..., 0]))
    return lat, lon


@dataclass
class Grid:
    """Grade regular em latitude/longitude, com os versores pré-computados."""

    n_lat: int = 91
    n_lon: int = 181

    def __post_init__(self):
        self.lat = np.linspace(-90.0, 90.0, self.n_lat)
        self.lon = np.linspace(-180.0, 180.0, self.n_lon)
        self.lat_mesh, self.lon_mesh = np.meshgrid(self.lat, self.lon, indexing="ij")
        # (n_lat, n_lon, 3) no referencial fixo à Terra -- calculado UMA vez
        self.points = geographic_to_ecef(self.lat_mesh, self.lon_mesh)
        self.unit = self.points / R_EARTH
        # pesos de área para médias sobre a esfera (cos da latitude)
        self.area_weight = np.cos(np.radians(self.lat_mesh))

    @property
    def shape(self):
        return (self.n_lat, self.n_lon)

    def equilibrium_field(self, bodies, t, theta0: float = 0.0, love: bool = False):
        """Maré de equilíbrio ``h(lat, lon)`` [m] somada sobre os corpos.

        ``bodies`` é uma sequência de pares ``(efeméride, GM)``. A soma é
        linear porque a maré é o gradiente de um potencial -- é isso que
        produz sizígia e quadratura sem nenhum tratamento especial.
        """
        total = np.zeros(self.shape)
        for eph, gm in bodies:
            pos = eci_to_ecef(eph.position(t), t, theta0)
            total += equilibrium_height(self.points, pos, gm, love=love)
        return total

    def area_mean(self, field) -> float:
        """Média ponderada por área -- deve dar ~0 para a maré de equilíbrio.

        A última coluna é descartada: a grade vai de -180 a +180 *inclusive*,
        então o meridiano de data aparece duas vezes e contá-lo em dobro
        enviesa a média.
        """
        field = np.asarray(field)[:, :-1]
        weight = self.area_weight[:, :-1]
        return float(np.sum(field * weight) / np.sum(weight))
