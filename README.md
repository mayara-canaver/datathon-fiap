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

Este repositório contém a implementação do fluxo de ingestão, tratamento, análise e deploy para o desafio do Datathon. Abaixo está o diagrama resumido do pipeline e suas etapas principais.

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

Breve instrução: siga a ordem das camadas (Bronze → Silver → Gold) e registre cada passo (versionamento de dados e experimentos). Arquivos importantes:

- `notebooks/01_ingestion_bronze.ipynb`
- `notebooks/02_ingestion_silver.ipynb`
- `notebooks/03_ingestion_gold.ipynb`

