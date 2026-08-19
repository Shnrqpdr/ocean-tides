# ocean-tides

Simulador de marés oceânicas em **dois níveis de física**, construído a partir
do formalismo newtoniano de Thornton & Marion, §5.5 "Ocean Tides" (pp. 198–204),
com visualizações animadas.

| nível | módulo | o que é |
|---|---|---|
| **equilíbrio** (Newton, 1687) | `forcing`, `response` | potencial de maré de grau 2 + resposta de bacia como oscilador forçado. É o que o PDF deriva. |
| **dinâmico** (Laplace, 1776) | `lte`, `bathymetry` | equações de águas rasas numa esfera girante, com batimetria real, Coriolis, atrito e auto-atração. |

A diferença entre os dois cabe em duas linhas:

```
onda de gravidade em água rasa:   c = √(gH) = √(9.81 × 4000)     ≈ 198 m/s
ponto sublunar, varrendo o solo:  v = 2πR / dia lunar (24h50min) ≈ 448 m/s
```

**A onda perde a corrida por 2.3×.** O oceano não consegue montar o bojo debaixo
da Lua. O que existe é uma onda forçada que reflete nas costas e gira, sob
Coriolis, em torno de pontos onde a maré simplesmente não sobe — os **pontos
anfidrômicos**. A teoria de equilíbrio não tem como produzi-los, e há um teste
que verifica isso.

## Instalação

```bash
uv sync --extra dev --extra data     # data traz xarray/netcdf4 p/ a batimetria
uv sync --all-extras                 # + numba, astropy, cartopy
```

Na primeira execução do modelo dinâmico, a batimetria ETOPO 2022 é baixada da
NOAA (~29 MB, subamostrada no servidor) e fica em cache em `data/`.

## Uso

**Maré de equilíbrio** — o modelo do livro:

```bash
uv run oceantides stations                        # estações e resposta em M2
uv run oceantides run --days 30 --out out.npz
uv run oceantides animate --mode polar --input out.npz   # Fig. 5-11a e 5-13
uv run oceantides animate --mode map   --input out.npz
uv run oceantides animate --mode globe --input out.npz
uv run oceantides bench                           # comparação de integradores
```

**Maré dinâmica** — as Equações de Maré de Laplace:

```bash
uv run oceantides lte --resolution 1.0 --constituent M2 --out lte_m2.npz
uv run oceantides chart    --input lte_m2.npz --save carta.png   # carta cotidal
uv run oceantides validate --input lte_m2.npz --plot             # vs marégrafos
uv run oceantides animate  --mode wave     --input lte_m2.npz
uv run oceantides animate  --mode currents --input lte_m2.npz
```

Gerar tudo de uma vez:

```bash
uv run python scripts/render_demos.py                    # 3 animações de equilíbrio
uv run --extra data python scripts/render_dynamics.py    # LTE + carta + validação
uv run --extra data python scripts/tune_friction.py      # recalibra a dissipação
uv run --extra data pytest                               # 247 testes
```

## O que emerge sem ser programado

Nenhuma frequência de maré, nenhum ponto anfidrômico e nenhum sentido de
rotação está codificado. Tudo sai das constantes orbitais e das equações:

* **M2 em 12 h 25.2 min**, S2 em 12 h exatas, batimento de sizígia/quadratura em
  14.77 dias, desigualdade diurna — verificados por FFT em `test_periods.py`.
* **Pontos anfidrômicos**, detectados pelo critério rigoroso do número de voltas
  da fase (±360° ao circundar o ponto), não por "amplitude pequena".
* **O sentido de rotação segue Coriolis**: 89% anti-horário no hemisfério norte,
  78% horário no sul. Sai do termo `f = 2Ω sin φ`.
* **No anfidromo a maré não sobe, mas a água corre**: elevação a 6% da mediana
  global, corrente a 102%.
* **O ciclo nodal de 18.6 anos**: com o nó orbital precessando, a declinação
  máxima da Lua oscila entre 18.3° e 28.6°, e as constituintes diurnas modulam
  ±18% (O1) — o que a análise harmônica corrige com os fatores nodais.

## Validação

Contra constantes harmônicas reais da API da NOAA CO-OPS:

* **ilhas oceânicas** (amostram oceano aberto — o alvo justo de um modelo de 1°):
  ver `docs/lte_validation.txt`;
* **estações costeiras** incluídas de propósito para *mostrar onde o modelo
  falha*, não para fingir que acerta.

A dissipação por maré interna é o único parâmetro ajustado, e foi calibrada
contra observação em `docs/friction_tuning.txt` — mínimo limpo em `r = 1e-5 s⁻¹`,
com o viés trocando de sinal exatamente ali.

## Estrutura

```
src/oceantides/
  constants.py    G, massas, elementos orbitais, precessões; classe BOOK
  forcing.py      Eq. 5.51 (exata) · Eq. 5.54 (expandida) · potencial · h_eq(ψ)
  integrators.py  rk4 · velocity-verlet · yoshida-4 · referência DOP853
  ephemeris.py    Lua/Sol: circular | kepler | nbody | astropy; nó e perigeu precessam
  grid.py         grade lat/lon, rotação da Terra, ponto sublunar
  response.py     oscilador forçado amortecido, RK4 vetorizado, admitância
  stations.py     marégrafos com parâmetros de bacia
  simulation.py   orquestração + spin-up do transiente
  bathymetry.py   ETOPO 2022, máscara de terra, limite CFL
  lte.py          Equações de Maré de Laplace, grade C, forward-backward
  harmonic.py     constituintes, números de Doodson, fatores nodais, análise
  validation.py   comparação com marégrafos da NOAA
  bench.py        comparação medida dos integradores
  viz/            polar_slice · world_map · globe3d · cotidal · tide_wave
docs/
  physics.md      derivação do PDF ao código; erratum do livro
  integrators.md  a resposta medida sobre RK4 vs. simpléticos vs. Kepler
  dynamics.md     por que o equilíbrio falha e o que as LTE acrescentam
  latex/          monografia: dinâmica clássica → força de maré → LTE →
                  formulação das melhorias; apêndice com a implementação
scripts/          render_demos · render_dynamics · tune_friction
```

## Três coisas que este projeto documenta

**1. Um erratum no livro.** A p. 203 afirma "1.46h = 0.83 m" para a maré de
sizígia, mas `1.46 × 0.54 = 0.79 m`. O fator e o valor impressos são
mutuamente inconsistentes. Ver `docs/physics.md` §4.

**2. Sobre o RK4.** A sugestão original de usar RK4 está certa exatamente onde
há uma EDO de verdade — a resposta do oceano, cujo amortecimento depende da
velocidade e portanto quebra os métodos simpléticos. Para as órbitas o RK4 é o
*pior* dos quatro implementados (erro de energia secular, medido: cresce ~9× em
10 anos). Mas simplético não implica mais preciso: Velocity-Verlet, de 2ª ordem,
fica ordens de grandeza pior. O vencedor é Yoshida-4. Na prática o pior erro de
maré de toda a tabela é **0.95 mm em 0.54 m**, então a escolha é numericamente
livre. Ver `docs/integrators.md`.

**3. Onde cada modelo falha.** O oscilador de 1 grau de liberdade acerta em
baixa latitude (Rio 1.13 m contra 1.20 m observados) e erra por 2–3× na Baía de
Fundy — e isso é física ausente, não ajuste ruim: os 16 m reais vêm da
co-oscilação com o sistema anfidrômico do Atlântico Norte, que só as LTE
descrevem. O modelo dinâmico, por sua vez, acerta em oceano aberto e erra na
costa, porque a plataforma continental não existe numa grade de 110 km. Ambos
os limites estão registrados em testes, não escondidos.

## Notas de implementação

* **Nada de campo global armazenado.** A maré de equilíbrio é forma fechada,
  recalculada por quadro (~2 ms). A solução das LTE guarda só os coeficientes
  harmônicos; o campo em qualquer instante sai deles em uma linha.
* **Os corpos é que giram, não a grade** — um vetor rotacionado em vez de 65 mil
  pontos.
* **O forçamento é uma função, não um array.** O RK4 avalia em `t + Δt/2`;
  pré-amostrar só em `t` derrubaria a ordem de 4 para 2. Há teste para isso.
* **Spin-up obrigatório** nos dois modelos: sem ele o transiente mascara o
  envelope de sizígia (bacia com Q=20 tem constante de tempo de 3.3 dias) e a
  onda global ainda não deu a volta no planeta.
* **Análise harmônica acumulada durante a integração**, em vez de guardar a
  série. É como os modelos operacionais funcionam.
* **Paleta validada, tema claro.** As quatro cores de série passam em todos os
  pares nos critérios de deficiência de visão de cor sobre a superfície clara;
  amplitude usa escala sequencial de um tom, elevação usa divergente ancorada em
  zero.
