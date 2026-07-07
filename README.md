 ![PPGI](img/ppgi-ufrj.png)

# SINALIZE

**Sin@lize** — Sistema preditivo de antecipação de falhas em transformadores de potência do Sistema Interligado Nacional (SIN/ONS)

## Resumo

Este trabalho apresenta o Sin@lize, ferramenta de manutenção preditiva baseada em aprendizado de máquina para antecipar falhas em transformadores da rede básica do SIN. O sistema avalia três abordagens — **Random Forest (RF)**, **XGBoost** e um **Ensemble RF+XGBoost** — em 13 horizontes temporais (30 a 90 dias), com base em 19 features de histórico de manutenção, falhas e utilização dos equipamentos.

O melhor resultado individual foi **AUC = 0,784 / F1 = 0,467 / Precisão = 0,778** no horizonte de **35 dias** (Ensemble e XGBoost), indicando que aproximadamente 3 em cada 4 alertas emitidos correspondem a falhas reais. O Ensemble obteve AUC médio de **0,677** ao longo dos 13 horizontes. A calibração via **Platt Scaling** reduziu o Brier Score médio em 13%, tornando as probabilidades diretamente interpretáveis para priorização de manutenções.

---

## Objetivos

- Prever falhas em transformadores da rede básica do SIN com antecedência de 30 a 90 dias
- Comparar RF, XGBoost e Ensemble em desempenho preditivo e qualidade probabilística
- Quantificar a incerteza das previsões via intervalos de confiança (95%)
- Demonstrar viabilidade de ML para manutenção preditiva em sistemas de energia em escala nacional

---

## Arquitetura

```
data/raw/
    ──► 01-preparar-dados  ──► data/gold/features_preditivo_{N}d.csv
                                        ──► 02-treinar-modelo ──► models/rf/
                                                                   models/xgb/
                                                                   models/rf_cal/
                                                                   data/gold/metricas_*.csv
                                        ──► 03-api  (POST /prever_falha)
```

### Fontes de dados (`data/raw/`)

| Arquivo | Descrição |
|---|---|
| `falhas.csv` | Eventos de falha (id, início, fim, duração, tipo) |
| `manutencao.csv` | Eventos de manutenção — SATRA/SAM (id, início, fim, duração) |
| `transformadores.csv` | Cadastro dos equipamentos (id, data de operação comercial, tensão, subestação) |
| `limites.csv` | Limite de potência por equipamento (MVA) |
| `utilizacao_transformadores/*.csv` | ~1.560 séries temporais de utilização — REGER/SCADA |

> **Dataset completo (dados brutos):** publicado no Zenodo — DOI [10.5281/zenodo.21010261](https://doi.org/10.5281/zenodo.21010261).
> Os arquivos de utilização (~6,7 GB) não ficam no repositório; baixe-os do Zenodo e salve em `data/raw/`
> (as séries de utilização em `data/raw/utilizacao_transformadores/`).

### Layout dos datasets (arquivos sem cabeçalho)

Todos os CSVs usam separador `;`, **não possuem linha de cabeçalho** e trazem datas no formato `AAAA-MM-DD[ HH:MM:SS]`. O `id_equipamento` é um UUID anonimizado. A ordem das colunas é:

| Arquivo | Colunas (na ordem) |
|---|---|
| `falhas.csv` | `id_equipamento`; `inicio` (datetime); `fim` (datetime); `duracao` (minutos); `tipo_falha` (ex.: `DEM`) |
| `manutencao.csv` | `id_equipamento`; `inicio` (datetime); `fim` (datetime); `duracao` (minutos) |
| `transformadores.csv` | `id_equipamento`; `data_entrada_operacao` (date); `tipo_arranjo_subestacao` (ex.: `BPRAUX`); `tensao_base_substacao` (kV, ex.: `230`) |
| `limites.csv` | `id_equipamento`; `limite` (ampère) |
| `utilizacao_transformadores/<id_equipamento>.csv` | `timestamp` (datetime, passo de 15 min); `valor` (carga em ampère). O nome do arquivo é o próprio `id_equipamento` |

### Features (19 features + target)

A data de referência é `data_máxima − janela_previsão`. Todos os cálculos usam apenas dados anteriores a essa data, evitando vazamento de dados.

| Categoria | Feature | Descrição |
|---|---|---|
| Equipamento | `idade_dias` | Idade na data de referência |
| | `limite_potencia` | Limite nominal de potência (MVA) |
| Manutenção | `num_manutencoes` | Total de manutenções realizadas |
| | `intervalo_medio_manut` | Intervalo médio entre manutenções (dias) |
| | `dias_desde_ultima_manut` | Dias desde a última manutenção |
| Falhas | `num_falhas_historico` | Total de falhas históricas |
| | `taxa_falhas_ano` | Falhas por ano |
| | `minutos_falha_historico` | Total de minutos em indisponibilidade por falha |
| | `taxa_minutos_falha_ano` | Minutos de falha por ano |
| | `dias_desde_ultima_falha` | Dias desde a última falha registrada |
| Utilização | `utilizacao_media` | Média da série de utilização |
| | `utilizacao_maxima` | Pico de utilização |
| | `utilizacao_minima` | Mínimo de utilização |
| | `utilizacao_desvio` | Desvio padrão da utilização |
| | `taxa_sobrecargas_ano` | Violações anualizadas do limite nominal |
| Tendência  | `p90_utilizacao` | Percentil 90 da utilização histórica |
| | `delta_utilizacao` | Diferença entre utilização média nos últimos 90d e o histórico total |
| | `utilizacao_tendencia_90d` | Coeficiente de tendência linear da utilização nos últimos 90d |
| | `dias_acima_80pct_limite` | Dias com utilização acima de 80% do limite |
| Target | `vai_falhar` | 1 se houve falha na janela de previsão, 0 caso contrário |

### Treinamento

- Apenas equipamentos com ao menos uma manutenção registrada (~1.559 transformadores)
- Divisão treino/teste: 70/30, estratificada por horizonte
- **RF:** SMOTE no treino + `class_weight='balanced'`; GridSearchCV (k=5), otimizando ROC-AUC
  - Grid: `n_estimators` ∈ {100, 200}, `max_depth` ∈ {10, 20, 30}, `min_samples_split` ∈ {2, 5}
- **XGBoost:** `scale_pos_weight` calculado por horizonte; GridSearchCV (k=5)
  - Grid: `n_estimators` ∈ {100, 200}, `max_depth` ∈ {4, 6, 8}, `learning_rate` ∈ {0.05, 0.1}
- **Ensemble:** média aritmética das probabilidades de RF e XGBoost
- **Calibração:** Platt Scaling (regressão sigmoidal) aplicada sobre os modelos RF

Artefatos gerados:

| Diretório / Arquivo | Conteúdo |
|---|---|
| `models/rf/` | 13 modelos RF serializados (`.pkl`) |
| `models/xgb/` | 13 modelos XGBoost serializados (`.pkl`) |
| `models/rf_cal/` | 13 modelos RF calibrados via Platt Scaling |
| `data/gold/metricas_performance.csv` | ROC-AUC, Precision, Recall, F1 por horizonte (RF) |
| `data/gold/metricas_xgboost.csv` | Métricas por horizonte (XGBoost) |
| `data/gold/metricas_ensemble.csv` | Comparativo RF vs XGBoost vs Ensemble (AUC) |
| `data/gold/metricas_consolidadas.csv` | Precision, Recall, F1, TP/FP/TN/FN — todos os modelos |
| `data/gold/metricas_calibracao.csv` | Brier Score antes/depois da calibração |
| `data/gold/thresholds_rf.csv` | Thresholds otimizados por horizonte (RF) |
| `data/gold/thresholds_xgb.csv` | Thresholds otimizados por horizonte (XGBoost) |
| `data/gold/importancias_features.csv` | Importância Gini por feature e horizonte (RF) |

### API REST

A API carrega todos os modelos RF dinamicamente e expõe três endpoints:

| Método | Endpoint | Descrição |
|---|---|---|
| `POST` | `/prever_falha` | Probabilidade, predição binária, desvio padrão e IC 95% por horizonte |
| `GET` | `/health` | Status da API e modelos disponíveis |
| `GET` | `/info` | Features requeridas e exemplo de requisição completo |

A incerteza é calculada agregando `predict_proba` de cada árvore individualmente (desvio padrão × 1,96 para IC 95%).

---

## Instalação e Uso

### 1. Instalar dependências

```bash
pip install -r requirements.txt
pip install imbalanced-learn  # não está no requirements.txt
```

### 2. Preparar dados

```bash
cd src
python 01-preparar-dados-preditivo.py
# Para horizontes específicos:
python 01-preparar-dados-preditivo.py 30,60,90
```

Gera `data/gold/features_preditivo_{N}d.csv` para cada horizonte.

### 3. Treinar modelos

> **Mac M1 16 GB:** antes de rodar, pause o OneDrive, execute `podman machine stop` e `sudo purge` para liberar RAM.

```bash
python 02-treinar-modelo-preditivo.py
# Para horizontes específicos:
python 02-treinar-modelo-preditivo.py 30,60,90
```

### 4. Executar a API

```bash
python 03-api-teste-modelo.py
```

API disponível em `http://localhost:5001`.

**Exemplo de requisição (19 features):**

```bash
curl -X POST http://localhost:5001/prever_falha \
  -H "Content-Type: application/json" \
  -d '{
    "id_equipamento": "TR-001",
    "idade_dias": 9893,
    "num_manutencoes": 6,
    "intervalo_medio_manut": 180,
    "num_falhas_historico": 2,
    "taxa_falhas_ano": 0.52,
    "minutos_falha_historico": 1255,
    "taxa_minutos_falha_ano": 35.9,
    "dias_desde_ultima_falha": 8,
    "dias_desde_ultima_manut": 107,
    "limite_potencia": 741,
    "utilizacao_media": 224.0,
    "utilizacao_maxima": 455.5,
    "utilizacao_minima": 0.0,
    "utilizacao_desvio": 76.4,
    "taxa_sobrecargas_ano": 0.0,
    "p90_utilizacao": 281.0,
    "delta_utilizacao": 38.7,
    "utilizacao_tendencia_90d": 0.85,
    "dias_acima_80pct_limite": 0,
    "periodos": [30, 35, 50]
  }'
```

**Exemplo de resposta:**

```json
{
  "id_equipamento": "TR-001",
  "previsoes": [
    {
      "periodo_dias": 35,
      "probabilidade": 0.98,
      "vai_falhar": 1,
      "desvio_padrao": 0.031,
      "intervalo_confianca": [0.919, 1.000]
    }
  ]
}
```

### 5. Gerar figuras do artigo (opcional)

```bash
cd temp
python gerar_figuras_artigo.py
```

Gera as 7 figuras em `temp/` utilizadas no artigo.

---

## Resultados

| Horizonte | Modelo | AUC | F1 | Precisão | Recall |
|:---:|---|:---:|:---:|:---:|:---:|
| **35d** | **Ensemble** | **0,784** | **0,467** | **0,778** | 0,333 |
| 35d | XGBoost | 0,784 | 0,452 | 0,700 | 0,333 |
| 50d | RF | 0,770 | 0,322 | 0,255 | 0,438 |
| *Média (13 horizontes)* | *Ensemble* | *0,677* | *0,292* | — | *0,359* |
| *Média (13 horizontes)* | *XGBoost* | *0,664* | *0,283* | — | *0,347* |
| *Média (13 horizontes)* | *RF* | *0,645* | *0,254* | — | *0,448* |

- Dataset: 1.559 transformadores de potência, período 2023–2024
- Desbalanceamento: 3,2% (30d) a 11,5% (90d) de falhas
- Thresholds otimizados individualmente por modelo e horizonte para maximizar F1

---

## Tecnologias

- **Python 3.11**
- **scikit-learn** — Random Forest, GridSearchCV, calibração, métricas
- **xgboost** — XGBoost
- **imbalanced-learn** — SMOTE
- **Flask** — API REST
- **Pandas / NumPy** — processamento de dados
- **joblib** — serialização de modelos
- **Matplotlib** — visualizações

---

## Trabalhos Futuros

- Stacking com meta-aprendiz supervisionado em substituição à média simples de probabilidades
- Integração com dados meteorológicos (temperatura, raios, vento) para features sazonais
- Integração com dados de monitoramento contínuo (temperatura do óleo, parâmetros de placa)
- Modelos de sobrevivência (*survival analysis*) para estimar diretamente o tempo até a próxima falha
- Extensão do pipeline para outros equipamentos do SIN (linhas de transmissão, bancos de capacitores, unidades geradoras)
