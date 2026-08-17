# Da maré de equilíbrio às Equações de Maré de Laplace

Este documento cobre o que foi acrescentado depois do primeiro commit: um
**modelo barotrópico global** que resolve as equações de águas rasas numa
esfera girante, com batimetria real. É o salto de 1687 para 1776 — de Newton
para Laplace.

---

## 1. Por que a teoria de equilíbrio tem que falhar

O PDF de Thornton & Marion deriva a maré de equilíbrio corretamente e obtém
`h = 0.54 m`. O que ele não diz é que essa teoria está **qualitativamente**
errada, não apenas imprecisa. O argumento é de uma linha:

```
onda de gravidade em água rasa:   c = √(gH) = √(9.81 × 4000)      ≈ 198 m/s
ponto sublunar, varrendo o solo:  v = 2πR / dia lunar (24h50min)  ≈ 448 m/s
```

**A onda perde a corrida por um fator de 2.3.** O oceano não consegue montar o
bojo debaixo da Lua porque a informação — a onda — não viaja rápido o
suficiente. Precisaria de um oceano de ~20 km de profundidade para acompanhar.

> A velocidade a comparar é a do ponto sublunar **relativa ao solo**, `2πR`
> dividido pelo dia lunar, e não `Ω·R ≈ 465 m/s` — esta é a velocidade de um
> ponto do equador no espaço inercial, que não é com quem a onda compete. A
> diferença é de 4% e não muda nada, mas o número certo é 448 m/s.

O que existe de verdade não é um bojo atrasado. É uma **onda forçada** que se
propaga, reflete nas costas e ressoa nos modos próprios de cada bacia. Sob a
força de Coriolis ela gira em torno de pontos onde a amplitude é exatamente
zero: os **pontos anfidrômicos**.

A teoria de equilíbrio não tem como produzi-los. Ela põe a maré sempre em fase
com o potencial, então a fase só assume dois valores (0° e 180°) e nunca
circula — há um teste que verifica exatamente isso
(`test_lte.py::test_equilibrium_theory_has_no_amphidromes`).

**Coriolis não é uma correção pequena.** A 45° de latitude,
`f = 2Ω sin φ = 1.03e-4 s⁻¹` contra `σ_M2 = 1.405e-4 s⁻¹`: a razão `f/σ` vale
**0.73**. É um termo de primeira ordem. No equador `f → 0`, o que ajuda a
explicar por que o modelo de bacia pontual do primeiro commit ia razoavelmente
bem no Rio e falhava em Fundy.

---

## 2. As equações resolvidas

Laplace (1776), em coordenadas esféricas, com `H(λ,φ)` a profundidade:

```
∂ζ/∂t = −1/(a cos φ) [ ∂(Hu)/∂λ + ∂(Hv cos φ)/∂φ ]

∂u/∂t =  f v − g/(a cos φ) ∂ζ'/∂λ − C_d |U| u / H − r u
∂v/∂t = −f u − g/a         ∂ζ'/∂φ − C_d |U| v / H − r v
```

com o desvio de equipotencial

```
ζ' = (1 − β) ζ − ζ_eq
```

| termo | o que representa | valor usado |
|---|---|---|
| `f = 2Ω sin φ` | Coriolis — sem ele não há anfidromo nem onda de Kelvin | — |
| `H(λ,φ)` | batimetria real, ETOPO 2022 | 5 arc-min → grade do modelo |
| `ζ_eq` | forçante astronômica (potencial de grau 2) | Lua + Sol, ou monocromática |
| `β` | **auto-atração e carga** (SAL): a própria maré deforma o geoide e a crosta | 0.09 |
| `C_d` | atrito de fundo quadrático — ~70% da dissipação total | 0.0025 |
| `r` | arrasto linear representando conversão para **maré interna** sobre topografia rugosa (~1 TW, 25–30% da dissipação) | calibrado, ver §4 |
| `(1+k₂−h₂)` | Terra elástica | 0.69 |

**Discretização.** Grade C de Arakawa (`ζ` no centro, `u` na face leste, `v` na
face norte), avanço *forward-backward*, atrito quadrático semi-implícito.
Longitude periódica; costas com fluxo normal nulo — são elas que refletem a
onda e criam os anfidromos.

**Duas escolhas que merecem nota.**

*Forçamento monocromático.* O modo padrão força **uma constituinte de cada
vez**, como TPXO e FES fazem. Com forçante astronômica completa, separar M2 de
S2 exige mais de 14.8 dias de registro pelo critério de Rayleigh; com forçante
monocromática a resposta também é monocromática, e bastam o spin-up e alguns
períodos.

*Análise harmônica embutida.* Em vez de guardar o campo a cada passo (GB de
dados), os produtos `ζ·cos(σt)` e `ζ·sin(σt)` são acumulados **durante** a
integração e o sistema normal é resolvido no fim. O mesmo vale para as
correntes. Guardam-se apenas os coeficientes; o campo em qualquer `t` é
reconstruído deles.

**Limite polar.** Acima de 86° tudo vira terra. É uma restrição numérica, não
física: a largura da célula encolhe com `cos φ`, e o passo estável tenderia a
zero no polo. A 86° e 1° de resolução a célula ainda tem 7.7 km, o que mantém
`dt ≈ 19 s`.

---

## 3. O que emerge sem ser programado

**Pontos anfidrômicos.** Detectados pelo critério rigoroso — o **número de
voltas da fase**. A definição correta não é "amplitude pequena" (isso também
acontece em baía fechada e em ruído costeiro), é que a fase percorre
exatamente 360° ao dar uma volta em torno do ponto. Somando as diferenças de
fase ao longo dos oito vizinhos, o total vale ±360° num anfidromo e 0 em
qualquer outro lugar.

**O sentido de rotação segue Coriolis.** Medido na rodada de produção a 1°,
que encontra **24 pontos anfidrômicos**:

| hemisfério | anfidromos | no sentido previsto |
|---|---|---|
| norte (anti-horário) | 8 | **75%** |
| sul (horário) | 13 | **85%** |

Isso não está em lugar nenhum do código. Sai do termo `f = 2Ω sin φ`.

A carta resultante (`media/carta_cotidal_M2.png`) reproduz os anfidromos reais
conhecidos: Atlântico Sul em ~(−25°, −12°), Atlântico Norte em ~(50°, −40°),
Índico em ~(0°, 62°), além dos máximos de amplitude no Atlântico Norte, na
plataforma patagônica e no noroeste da Austrália.

**A maré não sobe no anfidromo, mas a água corre.** Medido:

| grandeza no anfidromo | fração da mediana global |
|---|---|
| elevação | **6.4%** |
| corrente | **102%** |

A energia atravessa o ponto normalmente; o que não acontece ali é a maré subir.
Cuidado com a formulação: elevação e corrente estão em quadratura de *fase*,
não de amplitude — espacialmente as duas amplitudes são bem correlacionadas
(~0.8), porque plataforma rasa tem maré grande *e* corrente grande.

**Amplificação em água rasa.** As maiores amplitudes aparecem em plataforma, não
em oceano profundo, porque a onda desacelera (`c = √(gH)`) e se comprime — a lei
de Green. Há teste para isso.

---

## 4. Validação contra marégrafos

Constantes harmônicas reais da API da NOAA CO-OPS, comparadas com o modelo.

**Por que estações insulares.** Num modelo global de 1°, uma célula tem ~110 km:
portos, estuários e plataformas estreitas simplesmente não existem na grade.
Ilhas oceânicas isoladas amostram a maré de oceano aberto, que é o que um
modelo barotrópico grosseiro tem chance de acertar. As estações costeiras estão
incluídas deliberadamente para **mostrar onde o modelo falha**, não para fingir
que acerta.

Resultado da rodada de produção a 1° (tabela completa em
`docs/lte_validation.txt`):

| estação insular | observado | modelo | erro |
|---|---:|---:|---:|
| Hilo, Hawaii | 0.220 m | 0.219 m | **−0.4%** |
| Apra Harbor, Guam | 0.220 m | 0.216 m | −1.6% |
| Sand Island, Midway | 0.119 m | 0.111 m | −6.5% |
| Nawiliwili, Kauai | 0.149 m | 0.161 m | +8.3% |
| Kwajalein | 0.467 m | 0.422 m | −9.5% |
| Honolulu, HI | 0.171 m | 0.149 m | −13.0% |
| Wake Island | 0.277 m | 0.225 m | −18.7% |
| Pago Pago, Samoa | 0.377 m | 0.506 m | +34.2% |

**RMS de 5.2 cm sobre as oito ilhas**, contra uma amplitude média observada de
25.0 cm — ou seja, ~21% de erro relativo, sem nenhuma assimilação de dados.
Para referência, os modelos operacionais que assimilam décadas de altimetria
chegam a ~0.9 cm.

Nas estações costeiras o RMS é de **119 cm**, dominado por Anchorage (−75%): a
grade de 110 km simplesmente não tem o Cook Inlet. Curiosamente The Battery
(Nova York, −1.9%) e Key West (−9.3%) saem bem, porque estão em plataformas
largas o bastante para a grade enxergar.

### A máquina harmônica validada contra as tábuas oficiais

Independente do modelo dinâmico, o módulo `harmonic` foi verificado contra as
**previsões oficiais da NOAA**: dadas as constantes que a própria NOAA publica,
reconstruímos as tábuas de maré deles.

| estação | amplitude observada | RMS da diferença | correlação |
|---|---:|---:|---:|
| Honolulu | 0.72 m | 0.052 m | **0.991** |
| San Francisco | 2.11 m | 0.067 m | **0.994** |
| Boston | 3.69 m | 0.110 m | **0.995** |
| Key West | 0.70 m | 0.078 m | **0.990** |

Ajustando as previsões da NOAA com os nossos argumentos astronômicos,
recuperamos as constantes publicadas com amplitude exata a três casas decimais
e fase a menos de 1°:

| | M2 | S2 | N2 | K2 | K1 | O1 | P1 | Q1 |
|---|---|---|---|---|---|---|---|---|
| Δ fase | +0.0° | −0.2° | +0.8° | +0.4° | −0.1° | +0.1° | +0.4° | +0.8° |

O resíduo de 5–11 cm vem das ~29 constituintes que a NOAA usa e nós não
incluímos. Este teste foi o que revelou que as **correções de fase diurnas
estavam com o sinal invertido** — erro de exatamente 180° em K1, O1, P1 e Q1,
com as semidiurnas já exatas desde o início.

---

## 5. O que ainda falta — a escada até os modelos operacionais

Onde este projeto está, e o que separaria dele um modelo de verdade:

| nível | o que acrescenta | RMS em oceano profundo |
|---|---|---|
| 0 — equilíbrio (Newton) | potencial de grau 2, resposta estática | erro de forma funcional |
| 1 — **LTE barotrópicas** ← *este projeto* | Coriolis, batimetria, atrito, SAL | dezenas de cm |
| 2 — LTE de alta resolução | 1/16°, malha não estruturada, plataforma resolvida | ~10 cm |
| 3 — assimilação de altimetria | décadas de TOPEX/Jason corrigem amplitude e fase | **~1 cm** |

Modelos operacionais (FES2014/2022, TPXO9/10, GOT5, EOT20, DTU23) estão todos
no nível 3, e reportam ≈0.9 cm de RMS em oceano profundo, ≈5 cm em plataforma
continental e ≈6.5 cm em água costeira (Stammer et al. 2014, *Rev. Geophys.*).

Termos de física que este modelo ainda não tem:

* **maré interna resolvida** — aqui ela é só um arrasto linear calibrado;
  resolvê-la exigiria estratificação e um modelo baroclínico;
* **SAL completo** — a aproximação escalar `β = 0.09` erra até 20% em regiões de
  plataforma; o cálculo correto é uma convolução global (harmônicos esféricos)
  a cada passo;
* **maré atmosférica e barômetro invertido** — ~1 cm por hPa;
* **gelo marinho** no Ártico e Antártico;
* **termos não-lineares de advecção**, que geram M4/M6 em estuários (o modelo
  os produziria se a grade resolvesse os estuários).

Nada disso é obstáculo conceitual — é escala computacional e dados. O salto
qualitativo, o que separa "bojo que segue a Lua" de "onda que gira em torno de
anfidromos", já está feito.

---

## Fontes

* Stammer, Ray, Andersen, Arbic et al. (2014), "Accuracy assessment of global
  barotropic ocean tide models", *Reviews of Geophysics* —
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2014RG000450
* Egbert & Ray (2000), "Significant dissipation of tidal energy in the deep
  ocean inferred from satellite altimeter data", *Nature* —
  https://www.nature.com/articles/35015531
* Garrett (1972), "Tidal Resonance in the Bay of Fundy and Gulf of Maine",
  *Nature* — https://www.nature.com/articles/238441a0
* Lyard et al. (2021), "FES2014 global ocean tide atlas: design and
  performance", *Ocean Science* — https://os.copernicus.org/articles/17/615/2021/
* ETOPO 2022 Global Relief Model, NOAA NCEI —
  https://www.ncei.noaa.gov/products/etopo-global-relief-model
* NOAA CO-OPS harmonic constituents API —
  https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/
