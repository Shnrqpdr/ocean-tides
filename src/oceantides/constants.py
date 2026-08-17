"""Constantes físicas e elementos orbitais.

Valores do livro (Thornton & Marion §5.5) são preservados em ``BOOK`` para os
testes de reprodução; o resto do código usa os valores modernos (IAU/CODATA),
que diferem em ~0.1% e mudam ``h`` na terceira casa decimal.
"""

from __future__ import annotations

import math

# --- Constantes fundamentais (CODATA 2018 / IAU 2015) ------------------------

G = 6.67430e-11  # m^3 kg^-1 s^-2

# --- Corpos ------------------------------------------------------------------

M_EARTH = 5.97217e24  # kg
M_MOON = 7.342e22  # kg
M_SUN = 1.98892e30  # kg

R_EARTH = 6.371e6  # m, raio médio volumétrico
G_SURFACE = 9.80665  # m/s^2, gravidade padrão

# Produtos GM (mais precisos que G*M isoladamente)
GM_EARTH = G * M_EARTH
GM_MOON = G * M_MOON
GM_SUN = G * M_SUN

# --- Rotação da Terra --------------------------------------------------------

# Dia sideral = 23h 56m 4.0905s
T_SIDEREAL_DAY = 86164.0905  # s
OMEGA_EARTH = 2.0 * math.pi / T_SIDEREAL_DAY  # rad/s ~ 7.292115e-5

OBLIQUITY = math.radians(23.4392911)  # obliquidade da eclíptica

# --- Órbita da Lua (geocêntrica) ---------------------------------------------

A_MOON = 3.844e8  # m, semieixo maior
E_MOON = 0.0549  # excentricidade
I_MOON = math.radians(5.145)  # inclinação em relação à eclíptica
T_MOON_SIDEREAL = 27.321661 * 86400.0  # s, mês sideral
T_MOON_SYNODIC = 29.530589 * 86400.0  # s, mês sinódico (ciclo sizígia/quadratura)

# Precessões da órbita lunar. São lentas demais para afetar um mês simulado,
# mas governam efeitos de longo prazo que a análise harmônica precisa:
#
# * o **nó** regride em 18.613 anos. É o ciclo nodal: a inclinação da órbita
#   em relação ao equador oscila entre 18.3 e 28.6 graus, modulando a amplitude
#   das constituintes diurnas em ate +-18% (O1) e das semidiurnas em +-3.7% (M2).
#   Com o nó fixo, a efeméride fica travada num ponto do ciclo -- e, na
#   configuração inicial deste projeto, exatamente no máximo.
# * o **perigeu** avança em 8.847 anos, e é o que separa N2 e L2 de M2.
T_MOON_NODE = 18.612958 * 365.25 * 86400.0  # s, retrógrado
T_MOON_PERIGEE = 8.847353 * 365.25 * 86400.0  # s, prógrado

# --- Órbita da Terra em torno do Sol -----------------------------------------

A_EARTH_ORBIT = 1.495978707e11  # m, unidade astronômica
E_EARTH_ORBIT = 0.016708
T_YEAR = 365.256363 * 86400.0  # s, ano sideral

# --- Períodos de maré derivados ----------------------------------------------

# Dia lunar: a Terra precisa girar um pouco mais que 360 graus para reencontrar
# a Lua, porque ela avançou na órbita.
OMEGA_MOON = 2.0 * math.pi / T_MOON_SIDEREAL
T_LUNAR_DAY = 2.0 * math.pi / (OMEGA_EARTH - OMEGA_MOON)  # ~24h 50.5min
T_M2 = T_LUNAR_DAY / 2.0  # ~12h 25.2min -- o "12h 26min" do PDF
T_S2 = 86400.0 / 2.0  # 12h exatas, maré solar semidiurna
T_SPRING_NEAP = T_MOON_SYNODIC / 2.0  # ~14.765 d, batimento sizígia/quadratura

# --- Correção de Terra não-rígida --------------------------------------------

# O PDF observa (p. 204) que "Earth is not rigid, and it is also distorted by
# tidal forces". O fator diminutivo combina os números de Love de grau 2:
# k2 ~ 0.30 (potencial induzido), h2 ~ 0.61 (deslocamento do solo).
# A maré observada por um marégrafo preso ao solo é reduzida por este fator.
LOVE_K2 = 0.30
LOVE_H2 = 0.61
LOVE_FACTOR = 1.0 + LOVE_K2 - LOVE_H2  # ~0.69


class BOOK:
    """Valores exatos usados por Thornton & Marion no Exemplo 5.5.

    Mantidos para que os testes reproduzam os números impressos no PDF
    (h = 0.54 m etc.) sem ficarem reféns de atualizações das constantes.
    """

    G = 6.67e-11
    M_MOON = 7.350e22
    M_SUN = 1.989e30
    R_EARTH = 6.37e6
    D_MOON = 3.84e8
    D_SUN = 1.496e11
    G_SURFACE = 9.80
