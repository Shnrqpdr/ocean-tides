# Física: do PDF ao código

Referência: **Thornton & Marion, *Classical Dynamics of Particles and Systems*,
§5.5 "Ocean Tides", pp. 198–204.** Os números de equação abaixo são os do livro
e aparecem citados nos docstrings dos módulos correspondentes.

---

## 1. A força de maré (Eq. 5.48 → 5.51)

A superfície da Terra não é um referencial inercial: a Terra e a Lua orbitam o
centro de massa comum. O livro monta o problema num referencial inercial
`x'y'z'` e depois passa para o geocêntrico.

Força sobre uma parcela d'água de massa `m`, no inercial (Eq. 5.48):

```
m r̈'_m = −(G m M_E / r²) e_r − (G m M_m / R²) e_R
```

Aceleração do centro da Terra causada pela Lua (Eq. 5.49):

```
M_E r̈'_E = −(G M_E M_m / D²) e_D
```

Subtraindo, obtém-se a aceleração no referencial geocêntrico **não-inercial**
(Eq. 5.50):

```
r̈ = −(G M_E / r²) e_r − G M_m ( e_R/R² − e_D/D² )
     └── gravidade local ──┘   └──── força de maré ─────┘
```

O segundo termo é a **força de maré** (Eq. 5.51). Ela existe porque a Lua puxa
a água mais próxima com mais força do que puxa o centro da Terra, e o centro da
Terra com mais força do que a água do lado oposto. É uma *diferença* de
atrações, não uma atração.

### Convenção de sinais — o detalhe que mais causa erro

Pela Figura 5-10a do livro:

| símbolo | aponta de → para |
|---|---|
| `e_r` | centro da Terra → parcela |
| `e_R` | **Lua → parcela** |
| `e_D` | **Lua → centro da Terra** |

Inverter `e_D` produz um resultado com o sinal trocado e magnitude ~2× maior,
que parece plausível até se comparar com a Eq. 5.52. Em
[`forcing.py`](../src/oceantides/forcing.py) a Eq. 5.51 é escrita na forma
equivalente e autoexplicativa

```
a_T = G M (d − r)/|d − r|³ − G M d/|d|³     ("atração no ponto" − "atração no centro")
```

com `d` o vetor Terra → corpo, que é a convenção natural para efemérides.

---

## 2. A expansão em `r/D` (Eq. 5.52 → 5.54)

Como `r/D ≈ 0.017` para a Lua, expandir `(1 + r/D)⁻²` e reter o primeiro termo
não nulo dá, no ponto sublunar e em quadratura:

```
F_Tx = +2 G m M_m r / D³        (Eq. 5.52)   para fora, ao longo do eixo Terra–Lua
F_Ty = −  G m M_m r / D³        (Eq. 5.53)   para dentro, a 90°
```

e para um ponto qualquer a um ângulo `θ` (Eq. 5.54a/b):

```
F_Tx = 2 G m M_m r cos θ / D³
F_Ty = −  G m M_m r sin θ / D³
```

Em forma vetorial 3D, que é como o código implementa:

```
a_T(r) = (G M / D³) [ 3 (r · d̂) d̂ − r ]
```

Com `d̂ = x̂` isso devolve `(GM/D³)(2x, −y, −z)` — as Eq. 5.52 e 5.53
simultaneamente. O traço `2 − 1 − 1 = 0` reflete que o potencial de maré é
harmônico: não há massa no ponto considerado.

**Erro da expansão, medido:** entre 1.7% e 2.5% conforme o ângulo — exatamente
`O(r/D)`, como o livro afirma. Dobrar `r/D` dobra o erro
(`tests/test_forcing.py::test_error_scales_linearly_with_r_over_D`).

---

## 3. O potencial de maré (derivado — não está no PDF)

O livro trabalha só com forças. Para obter a *forma da superfície* precisamos do
potencial, que sai de integrar `F = −∇U` a partir das Eq. 5.54:

```
U_T/m = −(G M / 2D³) [ 3 (r · d̂)² − r² ] = −(G M r² / 2D³)(3 cos²ψ − 1)
```

onde `ψ` é o ângulo entre a parcela e o eixo Terra–corpo. Derivando de volta:
`−∂U/∂x = 2GMx/D³` e `−∂U/∂y = −GMy/D³` ✔ (verificado numericamente em
`tests/test_forcing.py::TestPotentialConsistency`).

A superfície do oceano em equilíbrio é uma **equipotencial**, logo `h = −U_T/(mg)`:

```
h_eq(ψ) = (G M r² / 2 g D³)(3 cos²ψ − 1)
        = (M/M_E)(r⁴/D³)(3 cos²ψ − 1)/2        [usando g = G M_E/r²]
```

A segunda forma elimina `G` e `g` do código. A diferença entre maré alta
(`ψ = 0`) e baixa (`ψ = 90°`) reproduz a **Eq. 5.55**:

```
h = 3 G M r² / (2 g D³)
```

O fator `(3cos²ψ − 1)` tem máximos em `ψ = 0` **e** `ψ = 180°`: os dois bojos
antipodais. É esse resultado contraintuitivo que Galileu não conseguiu explicar
(p. 198) e que produz **duas** marés altas por dia lunar, não uma.

---

## 4. Números reproduzidos

Calculados com as constantes exatas do livro (classe `BOOK` em `constants.py`),
travados em `tests/test_equilibrium.py`:

| grandeza | calculado | livro |
|---|---|---|
| `h` lunar (Eq. 5.55) | **0.5377 m** | 0.54 m ✔ |
| `h` solar | 0.2461 m | — |
| razão Sol/Lua (efeito de maré) | **0.4577** | 0.46 ✔ |
| razão Sol/Lua (atração direta) | 178 | ~175 ✔ |
| `r/D` | 0.0166 | 0.02 ✔ |
| período M2 | **12 h 25.2 min** | 12 h 26 min ✔ |
| erro da expansão | **1.7 – 2.5 %** | `O(r/D)` ✔ |

O contraste entre as duas primeiras razões é o coração da seção: **o Sol puxa
178 vezes mais forte que a Lua e ainda assim produz menos da metade da maré**,
porque o que gera maré é o *gradiente* do campo, que cai com `1/D³` em vez de
`1/D²`.

### Erratum do livro

A p. 203 afirma: *"The maximum tide, which occurs every 2 weeks, should be
1.46h = 0.83 m for the spring tides."*

Com `h = 0.54 m`, temos `1.46 × 0.54 = 0.79 m`, não 0.83 m. O fator e o valor
impressos são mutuamente inconsistentes (0.83/1.46 = 0.57 m, que não é o `h` do
próprio Exemplo 5.5). O teste
`test_equilibrium.py::test_spring_tide_height` trava o valor aritmeticamente
correto, **0.785 m**, e assinala explicitamente que 0.83 m não fecha.

---

## 5. Além do equilíbrio: a resposta dinâmica

O PDF encerra observando (p. 203) que as marés costeiras reais superam muito os
0.54 m, que *"local effects can be dramatic"* e que *"resonances can affect the
natural oscillation of the bodies of water"*. Também nota (Fig. 5-13) que a maré
alta não fica exatamente sobre o eixo Terra–Lua.

O modelo mínimo que captura as duas coisas é uma bacia como oscilador forçado
amortecido:

```
ḧ + 2γ ḣ + ω₀² h = ω₀² h_eq(t)      com  ω₀ = 2π/T₀,  Q = ω₀/2γ
```

Em regime permanente, na frequência `ω`:

```
A(ω) = ω₀² / √[(ω₀² − ω²)² + 4γ²ω²]          amplificação
δ(ω) = atan2(2γω, ω₀² − ω²)                   atraso de fase
```

* `ω ≪ ω₀` (bacia rígida): `A → 1`, `δ → 0` — segue a maré de equilíbrio.
* `ω ≈ ω₀` (ressonância): `A → Q`, `δ = π/2` — amplificação enorme.
* `ω ≫ ω₀` (bacia lenta): `A → 0`, `δ → π` — responde **abaixo** do equilíbrio.

O terceiro caso é real: o Taiti tem maré de ~0.3 m, *menor* que a de equilíbrio.

**Do atraso de fase para a Figura 5-13.** O padrão de maré é de grau 2, ou seja,
varre dois ciclos de fase a cada rotação relativa. Um atraso de fase `δ`
corresponde a um deslocamento **espacial** do bojo de `δ/2`
(`viz/common.py::phase_lag_to_angle`). É assim que o painel direito da animação
polar desenha o bojo adiantado em relação ao eixo Terra–Lua.

### Onde o modelo acerta e onde não

Medido em `tests/test_stations.py`:

| estação | modelado | observado |
|---|---|---|
| Rio de Janeiro | 1.11 m | 1.20 m |
| Santos | 1.15 m | 1.40 m |
| Belém | 2.99 m | 3.50 m |
| São Luís | 4.08 m | 6.60 m |
| **Burntcoat Head (Fundy)** | **5.2 m** | **16.0 m** |

A subestimação em Fundy **não é erro de ajuste**. São duas causas físicas:

1. A maré de equilíbrio *local* a 45° de latitude vale só ~66% da equatorial —
   o ponto nunca chega perto do ponto sublunar (medido:
   `test_high_latitude_equilibrium_forcing_is_weaker`).
2. Os 16 m reais vêm da co-oscilação da baía com o sistema anfidrômico do
   Atlântico Norte inteiro — um modo de bacia oceânica, não o forçamento local.

Reproduzir isso exigiria as **equações de maré de Laplace** (águas rasas numa
esfera girante), não um oscilador de um grau de liberdade. Os parâmetros `T₀` e
`Q` das estações são, portanto, parâmetros concentrados ajustados, e estão
documentados como tais em `stations.py`.

---

## 6. O que emerge sem ser programado

Nenhuma frequência de maré aparece no código — só constantes orbitais. Estas
saem da simulação e são verificadas em `tests/test_periods.py`:

* **M2 em 12 h 25.2 min** — meio dia lunar, não meio dia solar. A Terra precisa
  girar um pouco mais de 360° para reencontrar a Lua, que avançou na órbita.
* **S2 em 12 h exatas** — meio dia solar, por definição.
* **Batimento de sizígia/quadratura em 14.77 dias** — metade do mês sinódico,
  do batimento entre M2 e S2. Sizígia é ~2.7× a quadratura.
* **Desigualdade diurna** — as duas marés altas do dia diferem, porque a
  declinação da Lua torna os bojos assimétricos em relação ao equador
  (p. 204). No modo `circular`, que zera a inclinação, a assimetria some —
  confirmando a causa.
* **Marés perigeanas** — `D` varia de 363 625 km a 405 870 km, e como `h ∝ D⁻³`
  isso sozinho modula a amplitude em **1.39×**.

---

## 7. Terra elástica

A p. 204 observa que *"Earth is not rigid, and it is also distorted by tidal
forces"*. A correção usa os números de Love de grau 2: o marégrafo está preso a
um solo que também sobe, então a maré medida é reduzida por

```
1 + k₂ − h₂ ≈ 1 + 0.30 − 0.61 = 0.69
```

Disponível via `equilibrium_height(..., love=True)` ou `--love` na CLI.
Desligado por padrão, para reproduzir os números do livro (Terra rígida).
