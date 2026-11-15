Projeto: Análise automatizada de curvas de crescimento microbiano a partir de dados de leitor de placas.

Trilha: Pesquisa
Linguagem: Python 3.10+

1. Contexto e problema

Em laboratórios de microbiologia, leitores de placa registram a densidade óptica (OD) de dezenas de poços ao longo de horas. A extração dos parâmetros de crescimento (taxa máxima, fase lag, capacidade de suporte) costuma ser feita manualmente em planilhas, poço a poço. Isso é lento, pouco reprodutível, sujeito a erro e dificulta comparar condições e cepas.

Gargalo: não existe um fluxo padronizado, documentado e testado que transforme o arquivo bruto do leitor em uma tabela de parâmetros confiável.

2. Objetivo

Desenvolver uma aplicação de linha de comando em Python que, a partir de um CSV de leitor de placas e de uma tabela de metadados, ajuste modelos de crescimento a cada poço e produza uma tabela de parâmetros, gráficos de ajuste e um relatório de qualidade, de forma reprodutível e testável.

2.1 Dentro do escopo
Leitura e validação de CSV em formato largo (uma coluna por poço).
Correção de branco, transformação logarítmica e ajuste de modelos por poço.
Seleção do melhor modelo por AIC.
Exportação de tabelas e gráficos.
Testes automatizados com dados sintéticos.
2.2 Fora do escopo
Interface gráfica.
Leitura de formatos proprietários de fabricantes (somente CSV).
Análise estatística comparativa entre condições (testes de hipótese).
Uso de IA como componente da solução (a IA é usada apenas como ferramenta de desenvolvimento).

3. Entradas

3.1 Arquivo de medidas (--input)

CSV em formato largo: uma linha por instante de leitura e uma coluna por poço.

Coluna	Tipo	Descrição
time_h	número	Tempo em horas, não negativo e estritamente crescente
time	texto hh:mm:ss	Alternativa a time_h, para leitores que exportam o tempo como relógio (ex.: 01:30:00; as horas podem passar de 24, como em 25:00:00)

Exatamente uma das duas colunas de tempo deve estar presente. Quando for time, a leitura a converte para horas decimais (01:30:00 → 1,5 h) e passa a tratá-la como time_h no restante do pipeline. Formatos ambíguos (como hh:mm) e com dias (como 1.02:00:00) não são aceitos nesta versão. | A1, A2, ..., H12 | número | OD medida no poço, com o nome da coluna sendo o identificador do poço (uma coluna por poço) |

Exemplo:

csv
time_h,A1,A2,B1
0.0,0.052,0.051,0.050
0.5,0.055,0.053,0.050
1.0,0.061,0.058,0.051

Células vazias em uma coluna de poço são tratadas como medida ausente (o ponto é descartado e contado no relatório). Internamente, a leitura converte a tabela para o formato longo (well, time_h, od), de modo que o restante do pipeline não depende do formato do arquivo.

3.2 Arquivo de metadados (--metadata)

CSV com uma linha por poço:

Coluna	Tipo	Descrição
well	texto	Identificador do poço (deve corresponder ao nome de uma coluna do arquivo de medidas)
strain	texto	Cepa ou isolado
condition	texto	Condição experimental (meio, temperatura, tratamento)
is_blank	booleano	true se o poço é branco (meio sem inóculo)
3.3 Parâmetros opcionais
Parâmetro	Padrão	Descrição
--output-dir	results/	Pasta de saída
--models	logistic,gompertz	Modelos a ajustar
--min-growth	0.1 (ajustável)	Aumento mínimo de ln(OD/OD0) para considerar que houve crescimento
--smooth-window	3 (ajustável)	Janela da suavização por mediana móvel (1 desativa)

4. Processamento
Leitura e validação da coluna de tempo (time_h ou time, com conversão de hh:mm:ss para horas decimais), dos nomes das colunas de poço, dos tipos e da consistência entre colunas e metadados; conversão interna para o formato longo.
Correção de branco: subtrai de cada medida a média do OD das colunas dos poços-branco no mesmo instante de tempo. Se não houver poços-branco, usa o primeiro valor de cada poço como linha de base e registra um aviso.
Suavização leve por mediana móvel (opcional).
Transformação: y(t) = ln(OD(t) / OD0), em que OD0 é a média dos primeiros pontos do poço. Medidas com OD corrigido menor ou igual a zero são descartadas e contadas no relatório.
Detecção de ausência de crescimento: se max(y) < min-growth, o poço é marcado como no_growth e não é ajustado.
Ajuste dos modelos por mínimos quadrados não lineares (scipy.optimize.curve_fit), com chutes iniciais derivados dos dados:
Logístico (parametrização de Zwietering et al., 1990): y(t) = A / (1 + exp(4·μ/A·(λ − t) + 2))
Gompertz modificado (Zwietering et al., 1990): y(t) = A · exp(−exp(μ·e/A·(λ − t) + 1))
em que μ é a taxa máxima de crescimento específico (h⁻¹), λ é a duração da fase lag (h) e A é a amplitude assintótica em ln(OD/OD0).
Seleção de modelo por AIC (erros gaussianos): AIC = n·ln(RSS/n) + 2k.
Parâmetros derivados: capacidade de suporte K = OD0 · exp(A) e tempo de duplicação t_d = ln(2)/μ.

5. Saídas

Todas gravadas em --output-dir.

Arquivo	Conteúdo
parameters.csv	Uma linha por poço: well, strain, condition, model, mu_max, lag_h, A, K, doubling_time_h, r2, aic, status
plots/<well>.png	Dados observados e curva ajustada, com os parâmetros anotados
qc_report.md	Resumo: poços ajustados, sem crescimento, não convergidos, pontos descartados, avisos
run_info.json	Versão do programa, parâmetros usados, data e hash do arquivo de entrada

Valores possíveis de status: ok, no_growth, fit_failed, blank.

6. Casos de erro e tratamento

Situação	Comportamento esperado
Arquivo de entrada inexistente ou ilegível	Encerra com mensagem clara e código de saída diferente de zero
Nenhuma coluna de tempo (time_h ou time), ou as duas presentes	Encerra informando que exatamente uma é obrigatória
Valor de time fora do formato hh:mm:ss (minutos ou segundos maiores que 59, campos faltando, texto)	Encerra indicando a linha e o valor inválido
Nenhuma coluna de poço além de time_h	Encerra com mensagem clara
Valores não numéricos em time_h ou em uma coluna de poço	Encerra indicando linha e coluna
Tempo (após a conversão para horas) com valores repetidos, decrescentes ou negativos	Encerra com erro, pois o eixo do tempo é inválido
Células vazias em uma coluna de poço	Pontos descartados e contados no relatório
Coluna de poço sem entrada nos metadados (ou o inverso)	Aviso no relatório; o poço sem metadados é processado com campos vazios
Poço com menos de 6 pontos válidos (ajustável)	status = fit_failed, com motivo no relatório
OD corrigido menor ou igual a zero	Ponto descartado e contado no relatório
Poço sem crescimento	status = no_growth, sem ajuste
Ajuste não converge	status = fit_failed; o processamento dos demais poços continua
Duas colunas com o mesmo identificador de poço	Encerra com erro, pois os dados são ambíguos (a verificação usa o cabeçalho bruto, já que o pandas renomeia duplicatas silenciosamente)

Princípio geral: um poço problemático nunca interrompe a análise dos demais; erros de formato do arquivo interrompem a execução.

7. Critérios de aceite

Os critérios usam curvas sintéticas geradas com semente fixa e ruído gaussiano (desvio-padrão de 0,01 em OD) (ajustável).

Recuperação de parâmetros: em curvas geradas com parâmetros conhecidos, o erro relativo é de no máximo 5% para μ, 10% para λ e 5% para K.
Seleção de modelo: em curvas geradas com cada modelo, o AIC escolhe o modelo verdadeiro em pelo menos 80% das curvas (ajustável).
Robustez: poços sem crescimento, com OD negativo ou com dados insuficientes recebem o status correto e não interrompem a execução.
Reprodutibilidade: duas execuções com as mesmas entradas e parâmetros produzem parameters.csv idêntico.
Testes: pytest executa sem falhas em ambiente limpo, usando apenas dados sintéticos.
Documentação: o README permite reproduzir a análise de exemplo seguindo somente as instruções escritas.
Conversão de tempo: o mesmo conjunto sintético, escrito uma vez com time_h e outra com time em hh:mm:ss, produz parameters.csv idêntico. Casos de borda cobertos por teste: 00:00:00 → 0 h, 01:30:00 → 1,5 h, 25:00:00 → 25 h, e rejeição de 00:75:00 e de 01:30.

8. Requisitos não funcionais

Dependências registradas em requirements.txt (previstas: numpy, pandas, scipy, matplotlib, pytest).
Uso via linha de comando: python -m growthcurves.cli --input data/synthetic/plate.csv --metadata data/synthetic/metadata.csv.
Módulos separados por responsabilidade: io.py (leitura do formato largo, validação e conversão para o formato longo), models.py (funções dos modelos), fit.py (ajuste e seleção), plots.py (gráficos) e cli.py (interface).
Semente aleatória fixa na geração de dados sintéticos.

9. Dados, privacidade e uso de IA

Dados: o repositório contém apenas dados sintéticos. Nenhum dado real de laboratório, sensível ou confidencial é versionado nem enviado em prompts.
LGPD: os dados do projeto não envolvem pessoas; não há dados pessoais.
IA como ferramenta: usada para planejamento da especificação, geração de testes, refatoração e diagnóstico de erros, dentro dos limites definidos em AGENTS.md.
IA como componente: não se aplica.
Validação do código gerado por IA: todo código é revisado manualmente, coberto por testes com parâmetros conhecidos e conferido contra um ajuste independente feito em pelo menos uma curva.

10. Referências

Zwietering, M. H.; Jongenburger, I.; Rombouts, F. M.; van 't Riet, K. (1990). Modeling of the bacterial growth curve. Applied and Environmental Microbiology, 56(6), 1875–1881.