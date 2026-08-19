# Escolha do integrador — RK4 é a escolha certa?

A pergunta original era se Runge–Kutta de 4ª ordem seria adequado, ou se outro
método seria igualmente efetivo. A resposta curta: **depende de qual dos dois
subsistemas**, e é diferente para cada um. Este documento mede em vez de
afirmar — reproduza com `uv run oceantides bench`.

---

## Há dois subsistemas, com exigências opostas

### 1. Resposta dinâmica da bacia — RK4 é a escolha certa

```
ḧ + 2γ ḣ + ω₀² h = ω₀² h_eq(t)
```

* O amortecimento `2γḣ` **depende da velocidade**, e o sistema deixa de ser
  hamiltoniano: não há estrutura simplética a preservar, então os métodos
  simpléticos perdem a vantagem — e deixam de ser explícitos, porque a força
  passa a depender da velocidade que estão avançando.
* O sistema **não é rígido**: `Δt = 60 s` contra um período de 12.42 h dá ~745
  passos por ciclo, onde o RK4 é preciso e barato.
* Há forçamento externo dependente do tempo, e o RK4 o avalia em `t + Δt/2`,
  que é justamente de onde vem a 4ª ordem.

Verificado em `tests/test_response.py`: o RK4 integrado bate com a admitância
analítica `A(ω)` e `δ(ω)` com erro relativo abaixo de 2×10⁻³, e a ordem
observada no forçamento não-autônomo é > 3.5.

> **Armadilha:** se `h_eq` fosse pré-amostrada apenas nos instantes `t` em vez de
> passada como função, o RK4 cairia silenciosamente para 2ª ordem. Por isso
> `make_forcing` devolve um *callable*, e há um teste que trava a ordem.

### 2. Órbitas da Lua e do Sol — RK4 é a pior escolha das quatro

Problema conservativo integrado por meses ou anos. O RK4 não é simplético: o
erro de energia **cresce proporcionalmente ao tempo**. E como a maré vai com
`D⁻³`, qualquer desvio em `D` é amplificado 3× (`δh/h = 3·δD/D`).

---

## Resultado medido — 10 anos de órbita lunar

Saída bruta em [`integrators.txt`](integrators.txt).

| método | Δt [s] | \|dE/E\| | erro pos [m] | erro em h [%] | aval/passo | tempo [s] | secular? |
|---|---:|---:|---:|---:|---:|---:|---|
| RK4 | 600 | 3.65e-13 | 1.37e-01 | 1.26e-09 | 4 | 9.6 | **sim** |
| Velocity-Verlet | 600 | 1.15e-07 | 3.22e+05 | 4.87e-03 | 1 | 3.4 | não |
| Yoshida-4 | 600 | 3.86e-13 | 2.00e+00 | 1.22e-08 | 3 | 7.7 | não |
| RK4 | 1800 | 6.64e-11 | 2.05e+01 | 4.20e-07 | 4 | 3.2 | **sim** |
| Velocity-Verlet | 1800 | 1.03e-06 | 2.90e+06 | 4.38e-02 | 1 | 1.1 | não |
| Yoshida-4 | 1800 | 3.42e-11 | 1.64e+02 | 8.87e-07 | 3 | 2.5 | não |
| RK4 | 3600 | 2.10e-09 | 5.86e+02 | 1.77e-05 | 4 | 1.6 | **sim** |
| Velocity-Verlet | 3600 | 4.15e-06 | 1.16e+07 | 1.75e-01 | 1 | 0.6 | não |
| Yoshida-4 | 3600 | 5.49e-10 | 2.62e+03 | 1.42e-05 | 3 | 1.3 | não |

Ordem de convergência observada (deve bater com a declarada):

| método | declarada | observada |
|---|---|---|
| RK4 | 4 | **4.023** |
| Velocity-Verlet | 2 | **2.000** |
| Yoshida-4 | 4 | **3.998** |

### Secular vs. limitado — a medida que separa os métodos

Razão entre o erro de energia em 10 anos e em 1 ano, a `Δt = 3600 s`:

| método | razão | interpretação |
|---|---|---|
| RK4 | **8.9** | cresce quase linearmente com o tempo — **secular** |
| Velocity-Verlet | 0.9 | oscila dentro de um teto — **limitado** |
| Yoshida-4 | 0.9 | oscila dentro de um teto — **limitado** |

Outra forma de ver: o RK4 passa 39% do tempo estabelecendo um novo máximo de
erro; os simpléticos, apenas 3%.

---

## As três conclusões que a medição impõe

**1. A distinção simplético/não-simplético é real, e o RK4 fica do lado errado.**
O erro de energia do RK4 cresce sem limite; o dos simpléticos, não. Confirmado
em `tests/test_integrators.py::TestSymplecticity`.

**2. Mas simplético ≠ mais preciso — e essa nuance costuma ser esquecida.**
Velocity-Verlet é simplético e ainda assim fica de 2 000 a 300 000 vezes pior
que o RK4, porque é de 2ª ordem. **Nestas escalas de tempo a ordem do método
pesa mais que a simpleticidade.** A vantagem simplética está no *formato* do
erro, não na magnitude.

**3. Yoshida-4 é o vencedor entre os integradores propriamente ditos.**
Tem 4ª ordem *e* erro limitado, por 3 avaliações de força por passo contra 4 do
RK4 — ou seja, é ao mesmo tempo mais estável e mais barato. A `Δt = 3600 s` seu
erro de energia é 4× menor que o do RK4, e a diferença só cresce com o tempo.

---

## O que o projeto usa, e por quê

**Kepler analítico**, resolvendo `M = E − e sin E` por Newton–Raphson. Precisão
de máquina, deriva exatamente zero, custo `O(1)` por instante consultado (não
`O(n)` por passo), e permite avaliar a posição em *qualquer* `t` — necessário
porque o RK4 da bacia pede a efeméride em `t + Δt/2`.

Os outros três continuam implementados e selecionáveis
(`--ephemeris nbody`), porque é o que torna a comparação acima reproduzível.

---

## E o veredito sobre a sugestão original?

**O pior erro de maré da tabela inteira é 0.18%** — cerca de **1 mm** numa maré
de 0.54 m — e vem do Velocity-Verlet no passo mais grosseiro. O RK4 fica em
2×10⁻⁵ %, ou seja, abaixo de um décimo de micrômetro.

Ou seja: para esta aplicação a escolha do integrador é numericamente livre. **A
sugestão original de RK4 funcionaria perfeitamente**, e nada no resultado físico
mudaria. O Kepler analítico foi adotado por ser mais simples e exato, não por
necessidade numérica; e a análise acima vale como o argumento de por que, num
problema orbital de escala maior, a resposta seria diferente.

---

## Métodos considerados e descartados

* **RKF45 / Dormand–Prince adaptativo.** Implementado só como referência de alta
  precisão (`integrators.reference_solution`, via SciPy DOP853 com
  `rtol=1e-12`). O passo adaptativo só compensaria se `D` variasse muito dentro
  de um passo, o que não é o caso: a excentricidade lunar é 0.055.
* **Leapfrog puro (posição/velocidade defasadas).** Equivalente ao
  Velocity-Verlet em precisão, mas com velocidades meio passo fora de fase, o
  que atrapalharia o cálculo de energia e a interpolação.
* **Métodos implícitos (Radau, BDF).** Feitos para sistemas rígidos. Nenhum dos
  dois subsistemas aqui é rígido: as escalas de tempo relevantes vão de 12 h a
  29.5 dias, uma separação de apenas ~60×.
