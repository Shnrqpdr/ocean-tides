# ocean-tides

Simulador de marés oceânicas construído a partir do formalismo newtoniano de
**Thornton & Marion, *Classical Dynamics of Particles and Systems*, §5.5
"Ocean Tides"** (pp. 198–204), com três visualizações animadas.

Nenhuma frequência de maré está codificada. Só constantes orbitais entram — M2
(12 h 25 min), S2 (12 h), o batimento de sizígia/quadratura (14.77 d) e a
desigualdade diurna **emergem** da simulação, e há testes que verificam isso.

## Instalação

```bash
uv sync --extra dev            # núcleo: numpy, scipy, matplotlib, pytest
uv sync --all-extras           # + numba, astropy, cartopy, xarray
```

## Uso

```bash
uv run oceantides stations                       # tabela de estações e resposta em M2
uv run oceantides run --days 30 --out out.npz    # simula
uv run oceantides animate --mode polar --input out.npz
uv run oceantides animate --mode map   --input out.npz
uv run oceantides animate --mode globe --input out.npz
uv run oceantides bench                          # comparação de integradores
uv run pytest                                    # 143 testes
```

Gravar em arquivo em vez de exibir: `--save saida.mp4` (ou `.gif` se não houver
`ffmpeg`), com `--dpi` e `--fps` ajustáveis. Reproduzir a hipótese simplificada
do livro (órbita circular, sem inclinação): `--ephemeris circular`. Terra
elástica: `--love`.

Para gerar os três vídeos de demonstração de uma vez, cada um na janela de tempo
que melhor revela seu fenômeno:

```bash
uv run python scripts/render_demos.py     # -> media/*.mp4
```

## As três visualizações

| modo | janela útil | o que mostra |
|---|---|---|
| `polar` | ~6 dias | Corte equatorial: campo de força da Eq. 5.54 (Fig. 5-11a) e os dois bojos, com o bojo dinâmico girado à frente do eixo Terra–Lua pelo atraso de fase (**Fig. 5-13**). Precisa de muitos quadros por dia: a escala aqui é o dia lunar de 24h50min. |
| `map` | ~30 dias | Mapa global de `h(lat, lon, t)` em escala divergente ancorada em zero, com a Terra girando sob os dois bojos, mais quatro marégrafos em *small multiples* onde o envelope de sizígia/quadratura aparece (batimento de 14.77 d). |
| `globe` | ~30 dias | Globo 3D deformado ao longo do ciclo de declinação da Lua (27.3 d). Com a Lua em declinação alta os bojos ficam assimétricos em relação ao equador — a desigualdade diurna da p. 204. |

## Estrutura

```
src/oceantides/
  constants.py    G, massas, elementos orbitais; classe BOOK com os valores do livro
  forcing.py      Eq. 5.51 (exata) · Eq. 5.54 (expandida) · potencial · h_eq(ψ)
  integrators.py  rk4 · velocity-verlet · yoshida-4 · referência DOP853
  ephemeris.py    Lua/Sol: circular | kepler | nbody | astropy
  grid.py         grade lat/lon, rotação da Terra, ponto sublunar/subsolar
  response.py     oscilador forçado amortecido, RK4 vetorizado, admitância
  stations.py     marégrafos com parâmetros de bacia
  simulation.py   orquestração, incluindo o spin-up do transiente
  bench.py        comparação medida dos integradores
  viz/            polar_slice · world_map · globe3d · common (paleta validada)
docs/
  physics.md      derivação completa, do PDF ao código; erratum do livro
  integrators.md  a resposta medida sobre RK4 vs. simpléticos vs. Kepler
scripts/
  render_demos.py gera os três vídeos em media/
```

## Três coisas que este projeto documenta e que valem a leitura

**1. O erratum do livro.** A p. 203 afirma "1.46h = 0.83 m" para a maré de
sizígia, mas `1.46 × 0.54 = 0.79 m`. O fator e o valor impressos são
mutuamente inconsistentes. Ver `docs/physics.md` §4.

**2. Sobre o RK4.** A sugestão de RK4 está certa exatamente onde há uma EDO de
verdade — a resposta do oceano, cujo amortecimento depende da velocidade e
portanto quebra os métodos simpléticos. Para as órbitas o RK4 é o *pior* dos
quatro métodos implementados: seu erro de energia é **secular** (medido: cresce
~9× em 10 anos, contra ~1× dos simpléticos). Mas a medição também mostra que
simplético não implica mais preciso — Velocity-Verlet, de 2ª ordem, fica ordens
de grandeza pior que o RK4. O vencedor é **Yoshida-4** (4ª ordem *e* limitado,
com 3 avaliações por passo contra 4). Na prática o pior erro de maré de toda a
tabela é **0.95 mm em 0.54 m**, então a escolha aqui é numericamente livre.
Ver `docs/integrators.md`.

**3. Onde o modelo falha.** Em baixa latitude ele chega perto do observado
(Rio 1.13 m contra 1.20 m; Belém 2.55 contra 3.50). Já a Baía de Fundy dá 7 m
contra 16 m reais — e isso é física ausente, não ajuste ruim: a maré de
equilíbrio *local* a 45° vale ~66% da equatorial, e os 16 m reais vêm da
co-oscilação com o sistema anfidrômico do Atlântico Norte, um modo de bacia
oceânica que só as equações de maré de Laplace descrevem. Registrado em
`tests/test_stations.py::TestKnownModelLimit`.

## Notas de implementação

* **O campo global não é armazenado.** É recalculado por quadro a partir da
  forma fechada (~2 ms). Guardar 4000 quadros de uma grade 181×361 custaria
  ~1 GB sem nenhum ganho.
* **Os corpos é que giram, não a grade.** Para obter `h(lat, lon)` no
  referencial terrestre, a Lua e o Sol são rotacionados para o referencial fixo
  à Terra — um vetor em vez de 65 mil pontos.
* **O forçamento é uma função, não um array.** O RK4 avalia `h_eq` em
  `t + Δt/2`; pré-amostrá-la só nos instantes `t` derrubaria silenciosamente a
  ordem do método de 4 para 2. Há um teste que trava isso.
* **Spin-up obrigatório.** Uma bacia com `Q = 20` tem constante de tempo de 3.3
  dias; sem integrar ~20 dias antes de `t = 0` e descartar, a saída começa com
  o oscilador subindo ao regime e o envelope de sizígia fica irreconhecível.
* **Paleta validada, tema claro.** As quatro cores de série passam em todos os
  pares nos critérios de deficiência de visão de cor sobre a superfície clara.
  No tema escuro nenhum quarto tom passa, então em vez de publicar um modo
  escuro que falha na acessibilidade, a superfície é fixa — ao contrário de uma
  página web, ela é conhecida de antemão numa figura matplotlib.
