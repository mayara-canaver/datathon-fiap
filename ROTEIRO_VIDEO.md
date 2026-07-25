# Roteiro de gravação — Vídeo Pitch (≤ 5 min)

Texto pronto para ler durante a gravação. Linguagem simples, direta, sem jargão
desnecessário. Cada bloco indica o que **falar** e o que **mostrar na tela**.

Antes de gravar, deixe abertos: terminal com a venv ativa, o `README.md`, e
rode `uvicorn src.api:app --port 8000` em outro terminal para a API já estar
no ar.

---

## 0:00–0:40 — Problema

**Mostrar na tela:** README aberto na seção "Objetivo".

**Falar:**

> "Um banco digital precisa decidir, para cada cliente, qual canal usar para
> oferecer um produto — por exemplo, ligar por celular ou por telefone fixo.
> Hoje isso costuma ser feito com uma regra fixa ou com testes A/B que
> demoram semanas. O problema é que isso desperdiça oportunidades: enquanto
> o teste roda, uma parte dos clientes continua recebendo o canal errado.
> Nosso projeto resolve isso com uma abordagem adaptativa, que aprende qual
> canal funciona melhor enquanto já está em operação."

---

## 0:40–1:20 — Abordagem

**Mostrar na tela:** pasta `notebooks/` e a tabela de pipeline no README
(Bronze → Silver → Gold).

**Falar:**

> "Usamos o dataset Bank Marketing, do Kaggle, com cerca de 41 mil clientes
> de campanhas reais de um banco. Organizamos os dados em três camadas:
> Bronze, com o dado bruto; Silver, já limpo e tipado; e Gold, pronta para o
> modelo. Um cuidado importante: removemos a coluna `duration`, porque ela só
> é conhecida depois que a ligação termina — se a mantivéssemos, o modelo
> estaria 'trapaceando', usando uma informação do futuro."

---

## 1:20–2:20 — Modelo

**Mostrar na tela:** tabela de resultados do `05_baseline_bandit.ipynb` ou a
tabela "Baseline × Bandit" do README.

**Falar:**

> "Comparamos duas políticas. A primeira é o baseline: a regra fixa que o
> banco usa hoje, sempre ligar por telefone fixo — ela converte cerca de
> 7%. A segunda é o Thompson Sampling, um algoritmo bayesiano que testa os
> dois canais e vai concentrando as tentativas no que converte mais. Ele
> chegou a quase 13% de conversão, um ganho de mais de 5 pontos percentuais
> sobre o baseline — e isso sem precisar travar a decisão numa regra fixa
> desde o início. Registramos todos esses parâmetros e métricas no MLflow,
> para rastrear os experimentos."

---

## 2:20–4:20 — Demo ao vivo

**Mostrar na tela:** terminal e/ou `/docs` da API rodando.

**Falar (enquanto executa):**

> "Agora vou mostrar o serviço funcionando de verdade. Essa é a nossa API,
> que recebe os dados de um cliente e devolve o canal recomendado."

Rodar (ou clicar em "Try it out" em `/docs`):

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

**Falar (com o resultado na tela):**

> "Esse é o exemplo GS01 do nosso Golden Set: um estudante que já teve
> sucesso em contato anterior. A API recomenda `cellular`, com a
> probabilidade de conversão estimada. Testamos isso com 5 perfis diferentes
> de cliente e, em todos, a recomendação fez sentido com o que esperávamos
> do negócio."

---

## 4:20–5:00 — Fechamento

**Mostrar na tela:** trecho do README sobre arquitetura em nuvem.

**Falar:**

> "Para colocar isso em produção, a arquitetura seria simples: os dados
> ficariam num object storage como o S3, o pipeline rodaria em jobs
> gerenciados, e essa mesma API subiria em um container. Duas limitações que
> deixamos claras: essa é uma simulação offline, feita com um modelo que
> estima a recompensa de cada canal — não é um teste em produção real — e
> decisões sensíveis de oferta continuariam com aprovação humana. Como
> próximo passo, o ideal seria validar essa política com um teste A/B
> controlado antes de liberar para todos os clientes. Obrigado!"

---

## Checklist rápido antes de apertar "gravar"

- [ ] API rodando e respondendo em `/health` e `/recommend`
- [ ] Exemplo GS01 testado uma vez fora da gravação (evita erro ao vivo)
- [ ] README aberto nas abas: Objetivo, Baseline × Bandit, Arquitetura em nuvem
- [ ] Cronômetro visível ou app de gravação com timer, para não passar de 5 min
- [ ] Áudio testado (sem eco/ruído) antes de começar
