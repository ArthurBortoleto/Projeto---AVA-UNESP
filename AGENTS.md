# AGENTS.md

Instruções para ferramentas de IA (e para quem revisa o código) que trabalham neste repositório.
Leia este arquivo e o `README.md` antes de alterar qualquer coisa.

## O projeto

`growthcurves` é uma aplicação de linha de comando em Python (3.10+) que lê o CSV de um leitor de
placas (OD por poço ao longo do tempo) e uma tabela de metadados, ajusta modelos de crescimento
(logístico e Gompertz, parametrização de Zwietering et al., 1990) a cada poço e grava
`parameters.csv`, gráficos por poço, `qc_report.md` e `run_info.json`.

## Comandos

```bash
python -m pip install -r requirements.txt
python -m pytest                          # todos os testes (cerca de 30 s)
python -m growthcurves.synthetic          # gera data/synthetic/plate.csv e metadata.csv
python -m growthcurves.cli --input data/synthetic/plate.csv --metadata data/synthetic/metadata.csv
```

Rode os comandos na pasta principal (a que contém `growthcurves/` e `tests/`).

## Estrutura

| Arquivo | Responsabilidade |
|---|---|
| `growthcurves/io.py` | Leitura e validação do CSV largo, conversão de tempo, formato longo, metadados. Define `InputError` |
| `growthcurves/preprocess.py` | Correção de branco, suavização, transformação `ln(OD/OD0)` |
| `growthcurves/models.py` | Funções dos modelos (`logistic`, `gompertz`) |
| `growthcurves/fit.py` | Ajuste, chutes iniciais, AIC, status, K e tempo de duplicação |
| `growthcurves/plots.py` | Gráficos por poço |
| `growthcurves/report.py` | `parameters.csv`, `qc_report.md`, `run_info.json` |
| `growthcurves/synthetic.py` | Dados sintéticos com semente fixa |
| `growthcurves/cli.py` | Interface de linha de comando |
| `tests/` | Um `test_*.py` por módulo, mais testes de ponta a ponta |

Cada módulo tem uma responsabilidade; não misture etapas em um só arquivo.

## Convenções de código

- Erros de formato dos arquivos de entrada levantam `InputError`, com mensagem em português que indica
  a linha e o valor com problema. O `cli.py` captura e sai com código 1.
- Um poço problemático **nunca** interrompe os demais: ele recebe um `status` (`no_growth`, `fit_failed`)
  com o motivo, e a execução continua.
- Saídas determinísticas: mesmas entradas e parâmetros geram `parameters.csv` idêntico
  (6 algarismos significativos, final de linha `\n`, sem aleatoriedade fora dos dados sintéticos).
- Gráficos com a API `Figure` do matplotlib, sem `pyplot` (sem estado global nem janelas).
- Nenhuma chamada de rede e nenhum estado global no programa.
- Mensagens ao usuário, relatórios e documentação em português.

## Regras do enunciado que não devem mudar sem decisão do autor

- Fórmulas dos modelos e do AIC (`n·ln(RSS/n) + 2k`), `K = OD0·exp(A)`, `t_d = ln(2)/μ`.
- Nomes e ordem das colunas de `parameters.csv` e os valores de `status`: `ok`, `no_growth`, `fit_failed`, `blank`.
- Nomes dos arquivos de saída e opções da linha de comando e seus padrões
  (`--min-growth` 0,2, `--smooth-window` 3, `--min-points` 6, `--output-dir` `results/`).
- Escopo: sem interface gráfica, só CSV, sem testes de hipótese entre condições.

## Testes

- Usam **apenas dados sintéticos**, com semente fixa (`DEFAULT_SEED = 42`) e ruído gaussiano de
  desvio-padrão 0,01 em OD; o OD inicial das curvas simuladas é `DEFAULT_OD0 = 0.2`.
- Toda mudança de comportamento vem com teste, e o README é atualizado junto.
- Critérios de aceite (README, seção 7): recuperação de parâmetros, AIC escolhendo o modelo verdadeiro
  em pelo menos 80% das curvas de cada modelo, robustez, reprodutibilidade, conversão de tempo e
  conferência contra um ajuste independente (`tests/test_independent_fit.py`).

## Limites do uso de IA

A IA é **ferramenta de desenvolvimento**, nunca componente da solução.

Permitido:

- planejar e revisar a especificação;
- gerar e melhorar testes;
- refatorar código;
- diagnosticar erros a partir de mensagens de erro e trechos de código.

Não permitido:

- chamar APIs de IA a partir do programa ou depender de IA em tempo de execução;
- receber, gerar ou versionar dados reais de laboratório, sensíveis ou confidenciais (só dados sintéticos);
- enfraquecer testes ou mudar tolerâncias apenas para que passem; se um critério precisa mudar, o motivo
  é registrado no README;
- afirmar que testes passam sem tê-los executado;
- ampliar o escopo ou adicionar dependências sem registrá-las em `requirements.txt` e no README.

Quando o enunciado for ambíguo, a IA explica a interpretação escolhida em vez de assumi-la em silêncio.

## Validação do código gerado por IA

1. Revisão manual pelo autor de todo o código.
2. Testes com parâmetros conhecidos (curvas simuladas com μ, λ e A conhecidos).
3. Conferência contra um ajuste independente (evolução diferencial) em curvas do exemplo.
4. `python -m pytest` passando em ambiente limpo, e o README reproduzindo a análise de exemplo.

## Dados e privacidade

O repositório contém apenas dados sintéticos. Os dados do projeto não envolvem pessoas e não há dados
pessoais (LGPD). Nenhum dado real é enviado em prompts.