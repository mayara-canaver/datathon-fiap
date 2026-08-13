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

## Governança e uso de dados

- **Base legal / natureza dos dados**: dataset público e anonimizado do Kaggle, sem dados reais de clientes, identificadores, patrimônio, renda, gênero, raça ou regras comerciais privadas. Uso didático/demonstrativo, sem implicações de compliance sobre dados pessoais reais.
- **Finalidade**: simular uma política de decisão de canal de contato (`cellular` vs `telephone`) em campanhas de marketing bancário, para fins de estudo de algoritmos adaptativos (multi-armed bandit).
- **Minimização**: apenas as colunas necessárias para o modelo de recompensa seguem até a camada Gold; `duration` (vazamento temporal) e `default` (pouco informativa) são descartadas ainda no tratamento.
- **Retenção**: o controle de versão mantém somente metadados, modelo treinado e políticas agregadas (`artifacts/`); as camadas `data/bronze`, `data/silver` e `data/gold`, com os dados linha a linha, não são versionadas (`.gitignore`) e podem ser recriadas a qualquer momento a partir do Kaggle.
- **Humano no loop**: em um cenário real, decisões sensíveis de oferta manteriam aprovação humana e limites de exploração antes de impactar o cliente final — a política adaptativa aqui é um apoio à decisão, não uma decisão autônoma.

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
5. `notebooks/05_baseline_bandit.ipynb` — baseline × Thompson Sampling
6. `notebooks/06_evaluation_golden_set.ipynb` — Golden Set (5 clientes)

As camadas `data/bronze`, `data/silver` e `data/gold` são geradas localmente (estão no `.gitignore`).

### Saídas esperadas após o S1

| Camada | Arquivo |
|--------|---------|
| Bronze | `data/bronze/bank_marketing_bronze.csv` |
| Silver | `data/silver/bank_marketing_silver.csv` |
| Gold | `data/gold/bank_marketing_gold.csv` |
| Gold train/test | `data/gold/bank_marketing_gold_train.csv`, `..._test.csv` |
| Metadata Gold | `data/gold/metadata.json` |

### Baseline × Bandit (S2)

Decisão modelada: **canal de contato** (`cellular` vs `telephone`).

| Política | Ideia | Conversão (simulação no test) |
|----------|-------|-------------------------------|
| Baseline legado | sempre `telephone` | ~7,1% |
| Random | braço aleatório | ~10,2% |
| Thompson Sampling | exploração bayesiana Beta–Bernoulli | ~12,8% |
| Melhor histórico | sempre `cellular` (teto empírico) | ~12,9% |

- Lift do Thompson Sampling vs baseline legado: **~+5,7 p.p.**
- Nota: o Thompson Sampling fica levemente abaixo do "melhor histórico" (teto empírico, que já nasce sabendo o braço vencedor) porque reserva parte do tráfego para exploração — isso é esperado, já que a política aprende essa preferência online, sem conhecer o melhor braço a priori.
- Artefatos: `artifacts/bandit_metrics.json`, `artifacts/thompson_policy.json`, `artifacts/reward_model.joblib`
- Código reutilizável: `src/bandit.py`

### Golden Set, API e MLflow (S3)

#### Golden Set (5 clientes)

Arquivo: [`artifacts/golden_set.json`](artifacts/golden_set.json) — também executável em `notebooks/06_evaluation_golden_set.ipynb`.

| ID | Persona | Oferta recomendada | Faz sentido? |
|----|---------|--------------------|--------------|
| GS01 | Estudante com sucesso prévio | `cellular` | Sim — histórico positivo |
| GS02 | Aposentado em 1º contato | `cellular` | Sim — canal com maior conversão |
| GS03 | Admin após falha prévia | `cellular` | Sim — corrige abordagem |
| GS04 | Blue-collar com muitos contatos | `cellular` | Sim — evita canal legado fraco |
| GS05 | Empresário (histórico telephone) | `cellular` | Sim — política adapta vs regra fixa |

#### API FastAPI

```bash
# na raiz do repositório, com a venv ativa
uvicorn src.api:app --reload --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Recomendação (exemplo GS01):

```bash
curl -s http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "exploit",
    "customer": {
      "age": 30, "job": "student", "marital": "single",
      "education": "professional course", "housing": "yes", "loan": "no",
      "month": "sep", "day_of_week": "tue", "campaign": 2, "pdays": 6,
      "previous": 1, "poutcome": "success", "emp.var.rate": -1.1,
      "cons.price.idx": 94.199, "cons.conf.idx": -37.5, "euribor3m": 0.88,
      "nr.employed": 4963.6, "never_contacted": 0, "previous_success": 1,
      "campaign_bucket": "few_contacts"
    }
  }'
```

Docs interativas: http://127.0.0.1:8000/docs

#### MLflow (tracking local)

```bash
python scripts/log_mlflow.py
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Abra http://127.0.0.1:5000 e revise o experimento `datathon-bandit` (conversão das políticas, lift e posteriors).

> No macOS, a porta 5000 pode estar em uso pelo AirPlay Receiver. Se a UI não abrir, desative em Ajustes do Sistema → Geral → AirDrop e Handoff → AirPlay Receiver, ou use outra porta (`--port 5001`).

### Arquitetura-alvo em nuvem (S4 — Etapa 6)

Em produção, o fluxo local seria mapeado para serviços gerenciados sem mudar a lógica do pipeline. Na **AWS**, o CSV/Kaggle (ou extratos internos anonimizados) entraria em um **Amazon S3** (camadas Bronze/Silver/Gold). O processamento (notebooks/jobs) rodaria em **AWS Glue**, **Amazon SageMaker Processing** ou um container em **ECS/Fargate**. Experimentos e artefatos do bandit ficariam no **MLflow** hospedado (Tracking Server + store em S3) ou no **SageMaker Model Registry**. A API FastAPI seria empacotada em imagem Docker e publicada via **Amazon ECS/Fargate** ou **AWS Lambda + API Gateway**, com logs/métricas no **CloudWatch**. Em **Azure**, o equivalente seria Blob Storage + Azure ML + Container Apps; em **GCP**, Cloud Storage + Vertex AI + Cloud Run. Em qualquer nuvem, decisões sensíveis de oferta manteriam **humano no loop** (aprovação/limites de exploração) e versionamento de dados/modelo para auditoria.

Diagrama lógico (opcional):

```text
Kaggle/origem → Object Storage (Bronze/Silver/Gold)
                      → Jobs de pipeline
                      → MLflow (params/métricas/artefatos)
                      → API containerizada → Canais digitais
                      → Monitoramento (logs, conversão, drift)
```

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
| 5 | `notebooks/05_baseline_bandit.ipynb` | Baseline × Thompson Sampling |
| 6 | `notebooks/06_evaluation_golden_set.ipynb` | Golden Set (5 casos) |
