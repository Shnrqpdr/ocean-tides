"""Força e potencial de maré (Thornton & Marion, Eq. 5.51-5.55).

Convenção de sinais do livro (Figura 5-10a), fácil de inverter por engano:

* ``e_R`` aponta da **Lua para a parcela** ``m``;
* ``e_D`` aponta da **Lua para o centro da Terra**.

Aqui trabalhamos sempre com ``body_pos``, o vetor **do centro da Terra para o
corpo perturbador** (Lua ou Sol), que é a convenção natural para efemérides.
A tradução para a do livro está anotada em :func:`tidal_accel_exact`.

Todas as funções fazem broadcasting NumPy: ``r`` tem forma ``(..., 3)`` e
``body_pos`` tem forma ``(3,)`` ou qualquer forma que faça broadcast com ``r``.
"""

from __future__ import annotations

import numpy as np

from .constants import GM_EARTH, R_EARTH

__all__ = [
    "tidal_accel_exact",
    "tidal_accel_expanded",
    "tidal_potential",
    "equilibrium_height",
    "tidal_range",
]


def _as_vec(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.shape[-1] != 3:
        raise ValueError(f"esperado vetor com última dimensão 3, recebido {a.shape}")
    return a


def tidal_accel_exact(r, body_pos, gm_body):
    """Aceleração de maré **exata**, sem expansão em ``r/D`` (Eq. 5.51).

    Escrita na forma fisicamente transparente "atração no ponto menos atração
    no centro da Terra"::

        a_T = GM (d - r)/|d - r|^3  -  GM d/|d|^3

    que é algebricamente idêntica à Eq. 5.51, ``-GM(e_R/R^2 - e_D/D^2)``, uma
    vez que ``e_R = (r - d)/|r - d|`` e ``e_D = -d/|d|``.

    É exatamente o segundo termo da Eq. 5.50: a diferença entre o puxão
    gravitacional do corpo no centro da Terra e na superfície.

    Parameters
    ----------
    r : array_like, forma (..., 3)
        Posição da parcela relativa ao centro da Terra [m].
    body_pos : array_like, forma (..., 3)
        Posição do corpo perturbador relativa ao centro da Terra [m].
    gm_body : float
        ``G * M`` do corpo perturbador [m^3/s^2].

    Returns
    -------
    ndarray, forma (..., 3)
        Aceleração de maré [m/s^2].
    """
    r = _as_vec(r)
    d = _as_vec(body_pos)

    sep = d - r  # da parcela para o corpo
    sep_norm = np.linalg.norm(sep, axis=-1, keepdims=True)
    d_norm = np.linalg.norm(d, axis=-1, keepdims=True)

    at_parcel = gm_body * sep / sep_norm**3
    at_center = gm_body * d / d_norm**3
    return at_parcel - at_center


def tidal_accel_expanded(r, body_pos, gm_body):
    """Aceleração de maré na expansão de 1ª ordem em ``r/D`` (Eq. 5.52-5.54).

    Forma vetorial::

        a_T = (GM/D^3) [ 3 (r . d_hat) d_hat  -  r ]

    Com ``d_hat`` ao longo de x e ``r = (x, y, z)`` isso devolve
    ``(GM/D^3)(2x, -y, -z)``, ou seja, exatamente as Eq. 5.52 e 5.53; e
    substituindo ``x = r cos(theta)``, ``y = r sin(theta)`` recupera as
    Eq. 5.54a/b.

    O erro relativo em relação a :func:`tidal_accel_exact` é ``O(r/D)``, ou
    ~2% para a Lua (o PDF cita ``r/D = 0.02`` na p. 200).
    """
    r = _as_vec(r)
    d = _as_vec(body_pos)

    d_norm = np.linalg.norm(d, axis=-1, keepdims=True)
    d_hat = d / d_norm
    radial = np.sum(r * d_hat, axis=-1, keepdims=True)  # r . d_hat
    return gm_body / d_norm**3 * (3.0 * radial * d_hat - r)


def tidal_potential(r, body_pos, gm_body):
    """Potencial de maré por unidade de massa [J/kg].

    O PDF só fornece forças; este potencial vem de integrar ``F = -grad U`` a
    partir das Eq. 5.54::

        U_T/m = -(GM / 2D^3) [ 3 (r . d_hat)^2 - r^2 ]
              = -(GM r^2 / 2D^3) (3 cos^2(psi) - 1)

    onde ``psi`` é o ângulo entre a parcela e o eixo Terra-corpo. Derivando de
    volta recupera-se ``(GM/D^3)(2x, -y, -z)``, confirmando a consistência com
    as Eq. 5.52/5.53.
    """
    r = _as_vec(r)
    d = _as_vec(body_pos)

    d_norm = np.linalg.norm(d, axis=-1)
    d_hat = d / d_norm[..., None]
    radial = np.sum(r * d_hat, axis=-1)
    r_sq = np.sum(r * r, axis=-1)
    return -gm_body / (2.0 * d_norm**3) * (3.0 * radial**2 - r_sq)


def equilibrium_height(r, body_pos, gm_body, love: bool = False):
    """Altura da maré de **equilíbrio** [m], relativa ao geoide sem perturbação.

    A superfície do oceano em equilíbrio é uma equipotencial, logo
    ``h = -U_T/(m g)``. Sobre a superfície (``|r| = R_EARTH``) isso é::

        h_eq(psi) = (GM R^2 / 2 g D^3) (3 cos^2(psi) - 1)

    e a diferença entre a maré alta (``psi = 0``) e a baixa (``psi = 90 graus``)
    reproduz a Eq. 5.55, ``3 G M r^2 / (2 g D^3)``.

    Parameters
    ----------
    love : bool
        Se ``True``, aplica o fator de Love ``(1 + k2 - h2) ~ 0.69`` para a
        Terra elástica, que o PDF menciona qualitativamente na p. 204. O padrão
        é ``False``, para reproduzir os números do livro (Terra rígida).
    """
    from .constants import LOVE_FACTOR

    g_local = GM_EARTH / R_EARTH**2
    h = -tidal_potential(r, body_pos, gm_body) / g_local
    return h * LOVE_FACTOR if love else h


def tidal_range(distance, gm_body, radius=R_EARTH, g=None):
    """Amplitude maré alta menos maré baixa da Eq. 5.55: ``3 G M r^2/(2 g D^3)``.

    É a forma fechada do Exemplo 5.5, que dá 0.54 m para a Lua.
    """
    if g is None:
        g = GM_EARTH / R_EARTH**2
    return 3.0 * gm_body * radius**2 / (2.0 * g * np.asarray(distance, dtype=float) ** 3)
