# Datathon FIAP — Fluxo do Projeto

Este projeto faz parte do Datathon e tem como objetivo construir uma solução end-to-end de Machine Learning Engineering para apoiar decisões adaptativas em canais digitais de uma instituição financeira.

A proposta é demonstrar como combinar análise de dados, modelagem, versionamento, avaliação e governança para personalizar ofertas e mensagens de forma mais eficiente do que regras fixas ou testes A/B longos.

## Objetivo

Criar uma solução que ajude a identificar padrões de comportamento do cliente, equilibrar exploração e explotação e sugerir a melhor ação para cada contexto, com foco em responsabilidade, observabilidade e tomada de decisão baseada em dados.

## Escopo

- ingestão e tratamento de dados
- preparação de dados para treinamento e avaliação
- construção de uma solução de recomendação/adaptação
- documentação de decisões, limitações e critérios de governança

## Base de dados (Kaggle)

- Dataset: [Bank Marketing — henriqueyamahata](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing)
- Arquivo utilizado: `bank-additional-full.csv` (~41.188 linhas, 21 colunas)
- Target: `y` (`yes` = converteu / `no` = não converteu)
- Observação: a coluna `duration` é removida na camada Gold por representar **data leakage**
- Metadados da origem: [`data/kaggle/metadata.json`](data/kaggle/metadata.json)

## Como executar

### Pré-requisitos

- Python 3.9+
- Conta Kaggle com API token (`~/.kaggle/kaggle.json` ou variáveis `KAGGLE_USERNAME` / `KAGGLE_KEY`) para o download na Bronze

### Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Pipeline de dados (ordem obrigatória)

Execute os notebooks nesta ordem:

1. `notebooks/01_ingestion_bronze.ipynb` — download Kaggle, validação de schema, persistência Bronze
2. `notebooks/02_ingestion_silver.ipynb` — limpeza, tipagem, deduplicação → Silver
3. `notebooks/03_eda.ipynb` — EDA, leakage, hipóteses para a Gold
4. `notebooks/04_ingestion_gold.ipynb` — feature engineering, split train/test, persistência Gold

As camadas `data/bronze`, `data/silver` e `data/gold` são geradas localmente (estão no `.gitignore`).

### Saídas esperadas após o S1

| Camada | Arquivo |
|--------|---------|
| Bronze | `data/bronze/bank_marketing_bronze.csv` |
| Silver | `data/silver/bank_marketing_silver.csv` |
| Gold | `data/gold/bank_marketing_gold.csv` |
| Gold train/test | `data/gold/bank_marketing_gold_train.csv`, `..._test.csv` |
| Metadata Gold | `data/gold/metadata.json` |

> Próximos sprints (ainda não implementados): baseline + Thompson Sampling, Golden Set, FastAPI, MLflow e parágrafo de arquitetura em nuvem.

## Pipeline (visão geral)

```text
                                         KAGGLE
                                           │
                                           ▼
                            Bank Marketing Dataset (.csv)
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           01 - INGESTÃO (BRONZE)                          │
├────────────────────────────────────────────────────────────────────────────┤
│ • Download do dataset                                                     │
│ • Leitura do CSV                                                          │
│ • Validação do schema                                                     │
│ • Registro da origem e versão dos dados                                   │
│ • Persistência da camada Bronze                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        02 - TRATAMENTO (SILVER)                            │
├────────────────────────────────────────────────────────────────────────────┤
│ • Conversão dos tipos                                                     │
│ • Limpeza de caracteres                                                   │
│ • Padronização das colunas                                                 │
│ • Remoção de duplicidades                                                 │
│ • Validação de domínio                                                    │
│ • Verificação de categorias                                               │
│ • Relatório de qualidade dos dados                                        │
│ • Persistência da camada Silver                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     03 - ANÁLISE EXPLORATÓRIA (EDA)                        │
├────────────────────────────────────────────────────────────────────────────┤
│ • Estatísticas descritivas                                                │
│ • Distribuição das variáveis                                              │
│ • Balanceamento do target                                                 │
│ • Correlação entre atributos                                              │
│ • Avaliação de outliers                                                   │
│ • Identificação de Data Leakage                                           │
│ • Formulação de hipóteses                                                 │
│ • Definição das transformações da camada Gold                             │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     04 - FEATURE ENGINEERING (GOLD)                        │
├────────────────────────────────────────────────────────────────────────────┤
│ • Remoção da variável duration (Data Leakage)                             │
│ • Criação da feature never_contacted                                      │
│ • Criação da feature previous_success                                     │
│ • Criação da feature campaign_bucket                                      │
│ • Seleção das variáveis finais                                            │
│ • Split train/test estratificado                                          │
│ • Persistência da camada Gold                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│      BASELINE FIXO           │              │  MODELO ADAPTATIVO (MAB)     │
├──────────────────────────────┤              ├──────────────────────────────┤
│ Política determinística      │              │ Thompson Sampling            │
│ ou melhor histórico          │              │ (ou Epsilon-Greedy / UCB)    │
│                              │              │                              │
│ Cálculo da taxa de conversão │              │ Aprendizado online           │
└──────────────────────────────┘              └──────────────────────────────┘
                      │                                         │
                      └────────────────────┬────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     05 - AVALIAÇÃO DOS MODELOS                             │
├────────────────────────────────────────────────────────────────────────────┤
│ • Comparação Baseline × Bandit                                             │
│ • Conversão média                                                         │
│ • Taxa de exploração                                                     │
│ • Regret (quando aplicável)                                               │
│ • Casos de teste (Golden Set)                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           06 - MLFLOW                                     │
├────────────────────────────────────────────────────────────────────────────┤
│ • Registro de parâmetros                                                  │
│ • Registro das métricas                                                   │
│ • Registro das versões dos modelos                                        │
│ • Comparação entre experimentos                                           │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         07 - SERVIÇO (FASTAPI)                             │
├────────────────────────────────────────────────────────────────────────────┤
│ Entrada: Dados do cliente                                                 │
│                                                                           │
│ → Pré-processamento                                                       │
│ → Modelo Adaptativo                                                       │
│ → Oferta recomendada                                                      │
│ → Probabilidade de conversão                                              │
└────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       08 - ARQUITETURA EM NUVEM                            │
├────────────────────────────────────────────────────────────────────────────┤
│ Kaggle → Storage → Pipeline → MLflow → API → Cliente                      │
└────────────────────────────────────────────────────────────────────────────┘
```

## Notebooks

| Ordem | Arquivo | Etapa |
|------|---------|-------|
| 1 | `notebooks/01_ingestion_bronze.ipynb` | Ingestão Bronze |
| 2 | `notebooks/02_ingestion_silver.ipynb` | Tratamento Silver |
| 3 | `notebooks/03_eda.ipynb` | EDA |
| 4 | `notebooks/04_ingestion_gold.ipynb` | Feature engineering Gold |
