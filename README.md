# growthcurves: análise automatizada de curvas de crescimento microbiano

Aplicação de linha de comando em Python que lê o arquivo bruto de um leitor de placas
(CSV com a densidade óptica, OD, de cada poço ao longo do tempo) e uma tabela de metadados,
ajusta modelos de crescimento a cada poço e gera, de forma reprodutível e testável:

- uma **tabela de parâmetros** (taxa máxima de crescimento, fase lag, capacidade de suporte etc.);
- um **gráfico por poço** com os dados observados e a curva ajustada;
- um **relatório de qualidade** (poços ajustados, sem crescimento, com falha, pontos descartados, avisos);
- um **registro da execução** (versão, parâmetros, data e hash dos arquivos de entrada).

Trilha: Pesquisa. Linguagem: Python 3.10+.

## 1. Reproduzindo a análise de exemplo

Siga os passos na pasta principal do projeto (a que contém `growthcurves/` e `tests/`).

```bash
# 1. (opcional, recomendado) criar um ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell ou cmd)
source .venv/bin/activate       # Linux ou macOS

# 2. instalar as dependências
python -m pip install -r requirements.txt

# 3. gerar os dados sintéticos de exemplo (data/synthetic/plate.csv e metadata.csv)
python -m growthcurves.synthetic

# 4. rodar a análise
python -m growthcurves.cli --input data/synthetic/plate.csv --metadata data/synthetic/metadata.csv

# 5. rodar os testes
python -m pytest
```

No Windows, se `python` não for reconhecido, use `py` no lugar (por exemplo, `py -m pip install -r requirements.txt`).

**Versões testadas**: Python 3.12 (Linux) com numpy 2.5, pandas 3.0, scipy 1.18, matplotlib 3.11 e pytest 9.2, seguindo exatamente os passos acima em um ambiente virtual novo; e Python 3.15 no Windows. As versões mínimas de `requirements.txt` são estimadas a partir dos recursos usados e não foram testadas.

O passo 4 imprime algo como:

```
Análise concluída: 12 poços analisados (11 ok, 1 sem crescimento, 0 com falha no ajuste); 3 poços-branco.
Resultados em: results
```

e cria a pasta `results/` com `parameters.csv`, `qc_report.md`, `run_info.json` e `plots/`
(um PNG por poço analisado). Os valores do `parameters.csv` devem ficar próximos de:

| well | strain | condition | model | mu_max (h⁻¹) | lag_h (h) | K (OD) | status |
|---|---|---|---|---|---|---|---|
| A2 | S1 | meio_A | logistic | ≈ 0,51 | ≈ 3,0 | ≈ 1,47 | ok |
| B2 | S1 | meio_B | logistic | ≈ 0,31 | ≈ 5,2 | ≈ 1,21 | ok |
| C2 | S2 | meio_A | gompertz | ≈ 0,43 | ≈ 3,9 | ≈ 1,48 | ok |
| D2 | S2 | meio_B | gompertz | ≈ 0,26 | ≈ 6,2 | ≈ 0,99 | ok |
| D3 | S2 | meio_B | | | | | no_growth |
| H1 | | meio | | | | | blank |

Os dados de exemplo foram simulados com parâmetros conhecidos (por exemplo, A2: μ = 0,50 h⁻¹,
λ = 3,0 h, K ≈ 1,48), então dá para conferir a recuperação dos parâmetros. Os dígitos exatos
podem variar na última casa entre computadores.

## 2. Uso

```
python -m growthcurves.cli --input PLATE.csv --metadata METADATA.csv [opções]
```

| Opção | Padrão | Descrição |
|---|---|---|
| `--input` | (obrigatória) | CSV de medidas, formato largo |
| `--metadata` | (obrigatória) | CSV de metadados |
| `--output-dir` | `results/` | Pasta de saída (criada se não existir) |
| `--models` | `logistic,gompertz` | Modelos a ajustar, separados por vírgula |
| `--min-growth` | `0.2` | Aumento mínimo de ln(OD/OD0) para considerar que houve crescimento |
| `--smooth-window` | `3` | Janela da mediana móvel (inteiro ímpar; `1` desativa) |
| `--min-points` | `6` | Mínimo de pontos válidos por poço para tentar o ajuste (mínimo permitido: 4) |
| `--version` | | Mostra a versão |

O programa **não apaga** arquivos de execuções anteriores. Ao reutilizar uma pasta de saída,
`parameters.csv`, `qc_report.md` e `run_info.json` são sobrescritos, mas gráficos antigos
de poços que não existem mais continuam em `plots/`. Para uma saída limpa, use uma pasta nova.

## 3. Formato dos arquivos de entrada

### 3.1 Medidas (`--input`)

CSV em formato largo, separado por **vírgula**, com ponto decimal: uma linha por instante
de leitura e uma coluna por poço. O cabeçalho deve ter **exatamente uma** coluna de tempo:

- `time_h`: tempo em horas (número, não negativo e estritamente crescente); ou
- `time`: tempo como relógio `hh:mm:ss` (por exemplo `01:30:00` = 1,5 h; as horas podem passar de 24, como em `25:00:00`).
  Formatos ambíguos (`hh:mm`) e com dias (`1.02:00:00`) são rejeitados.

As demais colunas são os poços, e o nome da coluna é o identificador do poço:

```csv
time_h,A1,A2,B1
0.0,0.052,0.051,0.050
0.5,0.055,0.053,0.050
1.0,0.061,0.058,0.051
```

Células vazias em um poço são tratadas como medida ausente (o ponto é descartado e contado no relatório).
Números precisam ser decimais simples: `0,5` (vírgula decimal), `nan` e `inf` são rejeitados.
O arquivo deve estar em UTF-8 (com ou sem BOM).

### 3.2 Metadados (`--metadata`)

CSV com uma linha por poço:

| Coluna | Descrição |
|---|---|
| `well` | Identificador do poço (igual ao nome da coluna no arquivo de medidas) |
| `strain` | Cepa ou isolado |
| `condition` | Condição experimental (meio, temperatura, tratamento) |
| `is_blank` | `true` se o poço é branco (meio sem inóculo), senão `false` |

Um poço do arquivo de medidas sem linha nos metadados é analisado com `strain` e `condition`
vazios, e o relatório registra um aviso (e vice-versa). Um poço repetido nos metadados é erro.

## 4. Saídas

Todas gravadas em `--output-dir`.

| Arquivo | Conteúdo |
|---|---|
| `parameters.csv` | Uma linha por poço do arquivo de medidas, na ordem das colunas |
| `plots/<poço>.png` | Dados observados e curva ajustada, com os parâmetros anotados |
| `qc_report.md` | Resumo, pontos descartados, poços sem crescimento e com falha (com o motivo), avisos |
| `run_info.json` | Versão do programa, parâmetros usados, data (UTC) e SHA-256 dos arquivos de entrada |

Colunas de `parameters.csv`: `well, strain, condition, model, mu_max, lag_h, A, K, doubling_time_h, r2, aic, status`.

| `status` | Significado | Campos numéricos |
|---|---|---|
| `ok` | Ajuste concluído; `model` é o de menor AIC | preenchidos |
| `no_growth` | `max(y) < --min-growth`; não é ajustado | vazios |
| `fit_failed` | Poucos pontos válidos ou nenhum modelo convergiu (o motivo está no `qc_report.md`) | vazios |
| `blank` | Poço-branco, usado só na correção | vazios |

Os números saem com 6 algarismos significativos e o arquivo usa final de linha `\n`,
para que duas execuções com as mesmas entradas gerem `parameters.csv` idêntico.
Poços `no_growth` e `fit_failed` também ganham um gráfico, só com os pontos observados e o motivo no título.

## 5. Como a análise é feita

1. **Leitura e validação** (`io.py`): tempo, nomes de poços, tipos, consistência com os metadados; conversão interna para o formato longo (`well`, `time_h`, `od`).
2. **Correção de branco** (`preprocess.py`): subtrai de cada medida a média da OD dos poços-branco **no mesmo instante** (usando os brancos que tiverem medida naquele instante).
3. **Suavização leve** por mediana móvel centrada (janela 3 por padrão; nas bordas a janela encolhe).
4. **Transformação**: descarta medidas com OD corrigida ≤ 0 e calcula `y(t) = ln(OD(t) / OD0)`, em que OD0 é a média dos 3 primeiros pontos mantidos do poço.
5. **Ausência de crescimento**: se `max(y) < --min-growth`, o poço é `no_growth` e não é ajustado.
6. **Ajuste** (`fit.py`) por mínimos quadrados não lineares (`scipy.optimize.curve_fit`), com chutes iniciais derivados dos dados:
   - Logístico: `y(t) = A / (1 + exp(4·μ/A·(λ − t) + 2))`
   - Gompertz modificado: `y(t) = A · exp(−exp(μ·e/A·(λ − t) + 1))`

   em que μ é a taxa máxima de crescimento específico (h⁻¹), λ é a duração da fase lag (h) e A é a amplitude assintótica em ln(OD/OD0) (parametrização de Zwietering et al., 1990).
7. **Seleção de modelo** por AIC com erros gaussianos: `AIC = n·ln(RSS/n) + 2k`.
8. **Parâmetros derivados**: capacidade de suporte `K = OD0 · exp(A)` e tempo de duplicação `t_d = ln(2)/μ`.

**R² e AIC são calculados em y = ln(OD/OD0), não em OD.** Os gráficos também usam esse eixo.

## 6. Decisões de projeto e interpretações do spec

Pontos em que o enunciado deixa margem e que foram fixados assim:

- **Sem poços-branco**: a OD **não** é corrigida e um aviso é registrado; o início de cada poço serve de linha de base (OD0). Subtrair o primeiro valor de cada poço zeraria o primeiro ponto e deixaria o OD0 próximo do ruído.
- **OD0** = média dos **3** primeiros pontos mantidos (o enunciado não dá o número).
- **A suavização vem antes do descarte de OD ≤ 0** (ordem do enunciado): um pico negativo isolado é removido pela mediana em vez de descartado.
- **Instante sem nenhum branco medido**: o ponto é descartado, contado no relatório e gera um aviso.
- **`--min-growth` = 0,2.** Com ruído de 0,01 em OD, um poço sem crescimento oscila em ln(OD/OD0) em torno de ±0,1 (com OD0 = 0,2); um limite de 0,1 deixaria muitos poços planos passarem por crescimento.
- **`--min-points` = 6** é uma opção da linha de comando (o enunciado diz "ajustável").
- **Poços-branco aparecem no `parameters.csv`** com `status = blank`, para que a tabela tenha uma linha por poço do arquivo.
- **Limites do ajuste**: A ∈ [0,05·max(y), 10·max(y)], μ ∈ [10⁻⁶, 50] h⁻¹, λ ∈ [0, tempo final]; três pontos de partida determinísticos por modelo, mantendo o de menor soma de quadrados.
- **Um poço problemático nunca interrompe os demais**; erros de formato de arquivo interrompem a execução.

## 7. Testes e critérios de aceite

```bash
python -m pytest
```

Todos os testes usam apenas dados sintéticos, com semente fixa e ruído gaussiano de desvio-padrão 0,01 em OD.

| Critério | Onde é verificado |
|---|---|
| Recuperação de parâmetros | `tests/test_fit.py` (curvas simuladas, 18 combinações de parâmetros × 5 sementes × 2 modelos) |
| Seleção de modelo por AIC (≥ 80% por modelo) | `tests/test_fit.py` |
| Robustez (sem crescimento, OD negativa, poucos dados, sem metadados) | `tests/test_fit.py`, `tests/test_cli.py` |
| Reprodutibilidade (`parameters.csv` idêntico) | `tests/test_outputs.py`, `tests/test_cli.py` |
| Conversão de tempo (`time_h` × `time` dão o mesmo resultado; `00:00:00`, `01:30:00`, `25:00:00` aceitos; `00:75:00` e `01:30` rejeitados) | `tests/test_io_time.py`, `tests/test_cli.py` |
| Conferência contra um ajuste independente | `tests/test_independent_fit.py` |

Resultados medidos nas curvas simuladas (OD inicial 0,2; 90 curvas por modelo):

| Medida | Resultado |
|---|---|
| AIC escolhe o modelo verdadeiro | 90,0% (logístico), 95,6% (Gompertz) |
| Erro relativo de μ | mediana 1,8%; 86,7% das curvas dentro de 5% |
| Erro relativo de λ | mediana 1,5%; 94,4% das curvas dentro de 10% |
| Erro relativo de K | máximo 1,2% (todas as curvas dentro de 5%) |

**Ajuste independente**: quatro curvas do exemplo (duas por modelo) são ajustadas de novo por evolução diferencial
(`scipy.optimize.differential_evolution`), uma busca global sem chute inicial, que não tem relação com o
`curve_fit` usado pelo programa. Os dois métodos chegam aos mesmos μ, λ e A e à mesma soma de quadrados
(diferença relativa de μ, λ e A abaixo de 0,1%), o que indica que o ajuste do programa encontra o mínimo
global e não um mínimo local. Essa conferência usa um método numérico diferente, não outro programa.

**Como os limites de μ e λ são verificados**: o ruído torna impossível garantir 5% (μ) e 10% (λ) em *cada* curva.
Os testes exigem que o erro **mediano** fique dentro do limite e que pelo menos 80% das curvas estejam
dentro dele; para K o limite de 5% vale em todas as curvas. O OD inicial das curvas simuladas é 0,2
porque, com 0,1, o ruído relativo é grande demais e o AIC do logístico fica abaixo de 80%.

## 8. Erros e códigos de saída

| Código | Situação |
|---|---|
| `0` | Análise concluída (mesmo que alguns poços tenham `no_growth` ou `fit_failed`) |
| `1` | Erro nos dados de entrada, nos parâmetros ou na gravação; a mensagem sai em `stderr`, começando com `erro:` |
| `2` | Uso incorreto da linha de comando |

Erros de formato (arquivo inexistente, colunas de tempo ausentes ou duplicadas, `time` fora do formato
`hh:mm:ss`, valores não numéricos, tempo repetido ou decrescente, colunas de poço duplicadas, metadados
inválidos) **interrompem a execução sem criar nenhum arquivo de saída**, e a mensagem indica a linha e o
valor com problema.

### Problemas comuns

- `ModuleNotFoundError: No module named 'growthcurves'`: rode os comandos na pasta principal do projeto, a que contém `growthcurves/`.
- `cannot import name '__version__'`: falta o arquivo `growthcurves/__init__.py` (ou ele está sem a linha `__version__`).
- `OSError: [Errno 9] Bad file descriptor` ao gravar arquivos: o projeto está em uma pasta sincronizada pelo OneDrive, que pode travar a criação de arquivos. Mova o projeto para uma pasta fora do OneDrive (por exemplo `C:\projetos\`) ou pause a sincronização.

## 9. Estrutura do projeto

```
growthcurves/
├── io.py          # leitura e validação do CSV largo, conversão de tempo, formato longo, metadados
├── preprocess.py  # correção de branco, suavização, transformação ln(OD/OD0)
├── models.py      # logístico e Gompertz (Zwietering et al., 1990)
├── fit.py         # ajuste, chutes iniciais, AIC, status, K e tempo de duplicação
├── plots.py       # gráficos por poço
├── report.py      # parameters.csv, qc_report.md, run_info.json
├── synthetic.py   # gerador de dados sintéticos (semente fixa)
└── cli.py         # interface de linha de comando
tests/             # um arquivo test_*.py por módulo, mais testes de ponta a ponta
data/synthetic/    # plate.csv e metadata.csv de exemplo (python -m growthcurves.synthetic)
requirements.txt
pytest.ini
```

## 10. Dados, privacidade e uso de IA

- **Dados**: o repositório contém apenas dados sintéticos. Nenhum dado real de laboratório, sensível ou confidencial é versionado nem enviado em prompts.
- **LGPD**: os dados do projeto não envolvem pessoas; não há dados pessoais.
- **IA como ferramenta**: usada no planejamento da especificação, na geração de testes, na refatoração e no diagnóstico de erros. A IA não é componente da solução.
- **Validação do código gerado com IA**: todo código é revisado manualmente e coberto por testes com parâmetros conhecidos. A conferência contra um ajuste independente está descrita na seção 7.

## 11. Limitações

- Só lê CSV (sem formatos proprietários de fabricantes) e não tem interface gráfica.
- Não faz análise estatística comparativa entre condições (testes de hipótese).
- O ajuste é feito em ln(OD/OD0): curvas que não atingem o platô ou com OD0 muito ruidoso têm parâmetros menos confiáveis.
- Com ruído alto em relação ao OD inicial, o critério `--min-growth` perde sensibilidade; ajuste-o ao nível de ruído do seu leitor.

## 12. Referência

Zwietering, M. H.; Jongenburger, I.; Rombouts, F. M.; van 't Riet, K. (1990). Modeling of the
bacterial growth curve. *Applied and Environmental Microbiology*, 56(6), 1875–1881.