"""Equações de Maré de Laplace: o oceano como onda, não como bojo.

Este é o salto que separa a teoria de 1687 da de 1776. A maré de equilíbrio
supõe que o oceano assume instantaneamente a forma da equipotencial. Ele não
consegue, e o motivo é quantitativo:

* a onda de gravidade em água rasa viaja a ``c = sqrt(gH)``, ou ~198 m/s num
  oceano de 4000 m;
* o ponto sublunar varre o solo do equador a ``2 pi R / dia lunar`` ~ 448 m/s.

A onda é **fisicamente incapaz de acompanhar a Lua**. O que existe de verdade é
uma onda forçada que se propaga, reflete nas costas e ressoa nos modos da
bacia -- e que, sob a força de Coriolis, gira em torno de **pontos
anfidrômicos** onde a amplitude é nula. Nada disso aparece na teoria de
equilíbrio, e é por isso que ela erra a *forma funcional*, não só a magnitude.

Equações resolvidas (Laplace, 1776), com ``a`` o raio da Terra, ``H(lambda,phi)``
a profundidade, ``zeta`` a elevação e ``f = 2 Omega sin(phi)``::

    d(zeta)/dt = -1/(a cos(phi)) [ d(Hu)/d(lambda) + d(Hv cos(phi))/d(phi) ]

    du/dt =  f v - g/(a cos(phi)) d(zeta')/d(lambda) - C_d |U| u / H
    dv/dt = -f u - g/a           d(zeta')/d(phi)     - C_d |U| v / H

com o desvio de equipotencial

    zeta' = (1 - beta) zeta - zeta_eq

onde ``zeta_eq`` é a maré de equilíbrio astronômica (o mesmo potencial de grau 2
do resto do projeto) e ``beta`` é a aproximação escalar de **auto-atração e
carga** (SAL): a própria maré redistribui massa, deformando o geoide e a crosta.
Ignorar SAL erra a amplitude em até 20% globalmente.

Discretização: grade C de Arakawa (``zeta`` no centro, ``u`` na face leste,
``v`` na face norte), avanço temporal *forward-backward*, atrito quadrático
semi-implícito. Longitude periódica; costas como fronteiras de fluxo normal
nulo, que é o que produz as reflexões e portanto os anfidromos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import GM_MOON, GM_SUN, G_SURFACE, OMEGA_EARTH, R_EARTH
from .ephemeris import make_ephemeris
from .grid import eci_to_ecef, geographic_to_ecef

__all__ = ["LTEConfig", "LaplaceTidalModel"]


# Aproximação escalar de SAL. Ray (1998) e a prática corrente em modelos
# barotrópicos globais usam beta entre 0.085 e 0.12.
SAL_BETA = 0.09

# Arrasto de fundo quadrático adimensional, valor padrão em modelos de maré.
DRAG_COEFFICIENT = 0.0025

# Arrasto linear representando a conversão para maré interna sobre topografia
# rugosa. Egbert & Ray (2000) atribuem ~1 TW (25-30% da dissipação total) a
# esse mecanismo, que um modelo barotrópico não resolve explicitamente.
#
# O valor NÃO é derivável do modelo -- é calibrado contra observação. A varredura
# em `scripts/tune_friction.py`, a 1 grau e contra oito marégrafos insulares da
# NOAA, dá um mínimo limpo (RMS em cm da amplitude de M2):
#
#     r = 0        ->  36.1   (viés +32.9: sem dissipação suficiente, tudo ressoa)
#     r = 3e-6     ->  18.6   (viés +16.0)
#     r = 1e-5     ->   5.2   (viés  +0.1)  <-- ótimo, e o viés troca de sinal aqui
#     r = 2e-5     ->   9.2   (viés  -8.0)
#     r = 5e-5     ->  17.4   (viés -15.6: superamortecido)
#
# Para comparação, os modelos globais que assimilam altimetria chegam a ~0.9 cm.
INTERNAL_TIDE_DRAG = 1.0e-5  # 1/s


@dataclass
class LTEConfig:
    """Configuração do modelo.

    ``forcing`` escolhe entre dois regimes de trabalho:

    ``"astronomical"``
        Forçante completa de Lua e Sol a partir das efemérides. Fisicamente
        mais rica, mas separar M2 de S2 na análise harmônica exige, pelo
        critério de Rayleigh, mais de 14.8 dias de registro -- ou seja,
        rodadas longas.
    ``"constituent"``
        Forçante monocromática de uma constituinte só, na forma geodésica
        padrão. É como TPXO e FES operam: resolve-se cada constituinte
        separadamente. A resposta é monocromática, então bastam o spin-up e
        alguns períodos para extrair amplitude e fase com precisão.
    """

    resolution: float = 1.0
    days: float = 45.0
    spin_up_days: float = 8.0
    forcing: str = "astronomical"
    constituent: str = "M2"
    dt: float | None = None  # None -> CFL automático
    cfl_safety: float = 0.40
    bodies: tuple = ("moon", "sun")
    ephemeris_mode: str = "kepler"
    sal_beta: float = SAL_BETA
    drag: float = DRAG_COEFFICIENT
    internal_drag: float = INTERNAL_TIDE_DRAG
    min_depth: float = 20.0
    lat_limit: float = 86.0
    sample_every: int = 12  # passos entre acumulações da análise harmônica
    love: bool = True  # Terra elástica: o padrão aqui é ligado, ao contrário
    #                    da maré de equilíbrio, que reproduz o livro rígido
    snapshot_every: float | None = None  # segundos entre snapshots guardados


class LaplaceTidalModel:
    """Modelo barotrópico global de maré em grade C.

    Uso típico::

        model = LaplaceTidalModel(LTEConfig(resolution=1.0, days=45))
        result = model.run(constituents=["M2", "S2", "N2", "K1", "O1"])
        amp, phase = result.constituent("M2")
    """

    def __init__(self, config: LTEConfig | None = None, bathymetry=None, **kw):
        from .bathymetry import Bathymetry

        self.config = config or LTEConfig(**kw)
        cfg = self.config

        self.bathy = bathymetry or Bathymetry(
            resolution=cfg.resolution, min_depth=cfg.min_depth, lat_limit=cfg.lat_limit
        )
        self.n_lat, self.n_lon = self.bathy.shape
        self.lat, self.lon = self.bathy.lat, self.bathy.lon

        self._build_metric()
        self._build_masks()
        self._build_forcing()
        if self.config.forcing == "constituent":
            self._build_constituent_forcing()

        self.dt = float(cfg.dt) if cfg.dt else self.bathy.cfl_timestep(cfg.cfl_safety)

    # ------------------------------------------------------------------
    # Geometria
    # ------------------------------------------------------------------

    def _build_metric(self):
        res = np.radians(self.bathy.resolution)
        self.dlambda = res
        self.dphi = res

        phi_c = np.radians(self.lat)[:, None]  # centros
        phi_v = np.radians(self.lat + self.bathy.resolution / 2.0)[:, None]  # face norte

        self.cos_c = np.cos(phi_c)
        self.cos_v = np.cos(phi_v)
        self.f_c = 2.0 * OMEGA_EARTH * np.sin(phi_c)  # em pontos u (mesma latitude)
        self.f_v = 2.0 * OMEGA_EARTH * np.sin(phi_v)

        self.dx_c = R_EARTH * self.dlambda * self.cos_c  # largura zonal no centro
        self.dy = R_EARTH * self.dphi

    def _build_masks(self):
        mask = self.bathy.mask
        depth = self.bathy.depth

        # face leste da célula (j, i): só há fluxo se as duas células são água
        self.mask_u = mask & np.roll(mask, -1, axis=1)
        self.h_u = np.where(self.mask_u, 0.5 * (depth + np.roll(depth, -1, axis=1)), 0.0)

        # face norte: a última linha é o topo do domínio, sempre fechada
        mask_north = np.zeros_like(mask)
        mask_north[:-1] = mask[:-1] & mask[1:]
        self.mask_v = mask_north
        h_v = np.zeros_like(depth)
        h_v[:-1] = 0.5 * (depth[:-1] + depth[1:])
        self.h_v = np.where(self.mask_v, h_v, 0.0)

        self.mask = mask
        self.depth = depth
        # profundidade segura para divisões (atrito)
        self.h_u_safe = np.where(self.h_u > 0, self.h_u, 1.0)
        self.h_v_safe = np.where(self.h_v > 0, self.h_v, 1.0)

    def _build_forcing(self):
        lat_mesh, lon_mesh = np.meshgrid(self.lat, self.lon, indexing="ij")
        self.unit = geographic_to_ecef(lat_mesh, lon_mesh) / R_EARTH  # (n_lat, n_lon, 3)

        gm = {"moon": GM_MOON, "sun": GM_SUN}
        self.bodies = [
            (make_ephemeris(b, self.config.ephemeris_mode), gm[b])
            for b in self.config.bodies
        ]

        from .constants import LOVE_FACTOR

        # A Terra elástica reduz a forçante efetiva por (1 + k2 - h2) ~ 0.69.
        # Aqui é ligado por padrão, ao contrário da maré de equilíbrio: lá o
        # objetivo é reproduzir os números do livro (Terra rígida); aqui é
        # reproduzir o oceano.
        self.love_factor = LOVE_FACTOR if self.config.love else 1.0

    def _build_constituent_forcing(self):
        """Pré-calcula a forma espacial da constituinte escolhida.

        Formas geodésicas padrão, por espécie (grau 2 do potencial):

        * espécie 0 (longo período): ``A (3 sin^2(phi) - 1)/2``
        * espécie 1 (diurna):        ``A sin(2 phi) cos(sigma t + lambda)``
        * espécie 2 (semidiurna):    ``A cos^2(phi) cos(sigma t + 2 lambda)``

        O sinal ``+n*lambda`` faz o padrão migrar para **oeste** conforme o
        tempo avança, que é o sentido em que o ponto sublunar se desloca visto
        de um referencial preso à Terra.
        """
        from .harmonic import CONSTITUENTS

        con = CONSTITUENTS[self.config.constituent]
        phi = np.radians(self.lat)[:, None]
        lam = np.radians(self.lon)[None, :]

        if con.species == 0:
            shape = (3.0 * np.sin(phi) ** 2 - 1.0) / 2.0
        elif con.species == 1:
            shape = np.sin(2.0 * phi)
        else:
            shape = np.cos(phi) ** 2
        # o número de onda zonal é a própria espécie: 1 ciclo por volta para as
        # diurnas, 2 para as semidiurnas, 4 para as de águas rasas
        wave = con.species * lam

        self._con = con
        self._con_amp = con.amplitude * self.love_factor * np.broadcast_to(
            shape, (self.n_lat, self.n_lon)
        )
        self._con_phase = np.broadcast_to(wave, (self.n_lat, self.n_lon)).copy()

    def equilibrium_field(self, t: float) -> np.ndarray:
        """Maré de equilíbrio no instante ``t``, no referencial fixo à Terra.

        É a *forçante* do modelo, não o resultado -- a diferença conceitual
        central deste módulo.
        """
        if self.config.forcing == "constituent":
            return self._con_amp * np.cos(self._con.omega * t + self._con_phase)

        total = np.zeros((self.n_lat, self.n_lon))
        for eph, gm in self.bodies:
            d = eci_to_ecef(eph.position(t), t)
            dist = np.linalg.norm(d)
            d_hat = d / dist
            cos_psi = (
                self.unit[..., 0] * d_hat[0]
                + self.unit[..., 1] * d_hat[1]
                + self.unit[..., 2] * d_hat[2]
            )
            amp = gm * R_EARTH**2 / (2.0 * G_SURFACE * dist**3)
            total += amp * (3.0 * cos_psi**2 - 1.0)
        return total * self.love_factor

    # ------------------------------------------------------------------
    # Passo de tempo (forward-backward)
    # ------------------------------------------------------------------

    def step(self, state, t: float, dt: float):
        eta, u, v = state
        cfg = self.config

        # --- continuidade: eta avança com as velocidades antigas
        flux_u = self.h_u * u
        flux_v = self.h_v * v * self.cos_v

        div = (flux_u - np.roll(flux_u, 1, axis=1)) / (R_EARTH * self.cos_c * self.dlambda)
        dvdy = np.zeros_like(eta)
        dvdy[1:] = flux_v[1:] - flux_v[:-1]
        dvdy[0] = flux_v[0]
        div = div + dvdy / (R_EARTH * self.cos_c * self.dphi)

        eta = np.where(self.mask, eta - dt * div, 0.0)

        # --- desvio de equipotencial, já com SAL e com a forçante astronômica
        eta_eq = self.equilibrium_field(t + dt)
        eta_dev = (1.0 - cfg.sal_beta) * eta - eta_eq

        # --- momento zonal, usando o eta recém-atualizado
        v_at_u = 0.25 * (
            v + np.roll(v, 1, axis=0) + np.roll(v, -1, axis=1)
            + np.roll(np.roll(v, 1, axis=0), -1, axis=1)
        )
        dpdx = (np.roll(eta_dev, -1, axis=1) - eta_dev) / (R_EARTH * self.cos_c * self.dlambda)
        speed_u = np.hypot(u, v_at_u)
        rhs_u = u + dt * (self.f_c * v_at_u - G_SURFACE * dpdx)
        u = rhs_u / (
            1.0
            + dt * (cfg.drag * speed_u / self.h_u_safe + cfg.internal_drag)
        )
        u = np.where(self.mask_u, u, 0.0)

        # --- momento meridional, já com o u novo (esquema forward-backward)
        u_at_v = 0.25 * (
            u + np.roll(u, 1, axis=1) + np.roll(u, -1, axis=0)
            + np.roll(np.roll(u, 1, axis=1), -1, axis=0)
        )
        dpdy = np.zeros_like(eta)
        dpdy[:-1] = (eta_dev[1:] - eta_dev[:-1]) / (R_EARTH * self.dphi)
        speed_v = np.hypot(v, u_at_v)
        rhs_v = v + dt * (-self.f_v * u_at_v - G_SURFACE * dpdy)
        v = rhs_v / (
            1.0
            + dt * (cfg.drag * speed_v / self.h_v_safe + cfg.internal_drag)
        )
        v = np.where(self.mask_v, v, 0.0)

        return (eta, u, v)

    # ------------------------------------------------------------------
    # Integração + análise harmônica embutida
    # ------------------------------------------------------------------

    def run(self, constituents=None, progress=None):
        """Integra e extrai amplitude e fase de cada constituinte.

        A análise harmônica é acumulada **durante** a integração: em vez de
        guardar o campo inteiro a cada passo (GB de dados), acumulamos os
        produtos ``zeta cos(sigma t)`` e ``zeta sin(sigma t)`` e resolvemos o
        sistema normal no fim. É o mesmo princípio dos modelos operacionais.
        """
        from .harmonic import CONSTITUENTS, HarmonicSolution

        cfg = self.config
        names = list(constituents or ["M2", "S2", "N2", "K1", "O1"])
        sigma = np.array([CONSTITUENTS[n].omega for n in names])
        n_con = len(names)

        state = (
            np.zeros((self.n_lat, self.n_lon)),
            np.zeros((self.n_lat, self.n_lon)),
            np.zeros((self.n_lat, self.n_lon)),
        )

        dt = self.dt
        n_spin = int(round(cfg.spin_up_days * 86400.0 / dt))
        n_total = int(round(cfg.days * 86400.0 / dt))

        # acumuladores: 2 por constituinte (cosseno e seno), para elevação e
        # para as duas componentes de corrente já interpoladas ao centro
        acc = np.zeros((2 * n_con, self.n_lat, self.n_lon))
        acc_u = np.zeros((2 * n_con, self.n_lat, self.n_lon))
        acc_v = np.zeros((2 * n_con, self.n_lat, self.n_lon))
        normal = np.zeros((2 * n_con, 2 * n_con))
        n_samples = 0

        snapshots, snapshot_times = [], []
        snap_interval = (
            int(round(cfg.snapshot_every / dt)) if cfg.snapshot_every else None
        )

        for k in range(n_total):
            t = k * dt
            state = self.step(state, t, dt)

            if not np.isfinite(state[0]).all():
                raise RuntimeError(
                    f"instabilidade no passo {k} (t={t / 3600:.1f} h). "
                    f"Reduza dt (atual {dt:.1f} s) ou cfl_safety."
                )

            if k >= n_spin:
                if (k - n_spin) % cfg.sample_every == 0:
                    ts = (k + 1) * dt
                    basis = np.empty(2 * n_con)
                    basis[0::2] = np.cos(sigma * ts)
                    basis[1::2] = np.sin(sigma * ts)
                    acc += basis[:, None, None] * state[0]
                    # correntes trazidas das faces para o centro da célula
                    uc = 0.5 * (state[1] + np.roll(state[1], 1, axis=1))
                    vc = 0.5 * (state[2] + np.roll(state[2], 1, axis=0))
                    acc_u += basis[:, None, None] * uc
                    acc_v += basis[:, None, None] * vc
                    normal += np.outer(basis, basis)
                    n_samples += 1

                if snap_interval and (k - n_spin) % snap_interval == 0:
                    snapshots.append(state[0].copy())
                    snapshot_times.append((k + 1) * dt)

            if progress and k % max(1, n_total // 20) == 0:
                progress(k, n_total, state)

        # resolve o sistema normal, o mesmo para todas as células
        def solve(a):
            return np.linalg.solve(normal, a.reshape(2 * n_con, -1)).reshape(
                2 * n_con, self.n_lat, self.n_lon
            )

        coeffs = solve(acc)
        cu, cv = solve(acc_u), solve(acc_v)

        return HarmonicSolution(
            names=names,
            cos_coeff=coeffs[0::2],
            sin_coeff=coeffs[1::2],
            u_cos=cu[0::2],
            u_sin=cu[1::2],
            v_cos=cv[0::2],
            v_sin=cv[1::2],
            mask=self.mask,
            lat=self.lat,
            lon=self.lon,
            depth=self.depth,
            n_samples=n_samples,
            dt=dt,
            config=cfg,
            snapshots=np.array(snapshots) if snapshots else None,
            snapshot_times=np.array(snapshot_times) if snapshot_times else None,
        )

    def summary(self) -> str:
        return (
            f"{self.bathy.summary()}\n"
            f"passo de tempo: {self.dt:.1f} s  ·  "
            f"{int(self.config.days * 86400 / self.dt):,} passos para "
            f"{self.config.days:g} dias".replace(",", " ")
        )
