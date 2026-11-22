# Undersampling adaptativo para detecção de fraude com aprendizado federado (Non-IID)

Código do artigo **"Detecção de Fraudes de Cartão de Crédito com Aprendizado Federado: Uma abordagem de *undersampling* adaptativo para cenários Non-IID"** (Engenharia de Computação, CEFET-MG).

O projeto simula um cenário de aprendizado federado (FedAvg) com regressão logística treinada por SGD sobre a base *Credit Card Fraud Detection*. Os dados são divididos entre 5 clientes por uma distribuição de Dirichlet (3 níveis de heterogeneidade, α ∈ {0,1; 0,5; 1,0}). Cada cliente decide localmente se e quanto da classe majoritária descartar, suspendendo o descarte quando tem poucas fraudes. O código compara três estratégias, com duas variantes de ponderação da agregação (pré e pós-balanceamento):

- **Adaptativo** (τ = 10, r = 10, n_piso = 100);
- ***Undersampling* sem suspensão** (τ = 0);
- **Sem balanceamento**.

## Requisitos

- Python 3 (as versões exatas usadas nos experimentos ficam registradas em `results/export/config_e_ambiente.txt`)
- Dependências em `requirements.txt`

## Instalação

```bash
git clone https://github.com/arthur-lage/trabalho-machinelearning.git
cd trabalho-machinelearning

python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

## Dataset

O dataset não está no repositório por causa do tamanho. Baixe o `creditcard.csv` no Kaggle e coloque-o na pasta `data/`:

1. Acesse <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.
2. Baixe e extraia o arquivo `creditcard.csv`.
3. Salve-o em `data/creditcard.csv`.

## Como rodar

Execute sempre a partir da raiz do projeto, nesta ordem:

```bash
# 1. Roda os experimentos (5 sementes x 3 valores de alpha x 2 ponderações x 3 estratégias)
python src/run_all.py

# 2. Gera as tabelas e o histórico por rodada a partir dos resultados brutos
python scripts/extrair_resultados.py

# 3. Gera a figura de AUPRC por rodada (ponderação pré-balanceamento)
python scripts/gerar_grafico_auprc.py
```

Observações:

- O passo 1 leva alguns minutos, pois o treino federado de cada configuração é repetido para todas as sementes.
- O passo 2 reexecuta o código para gravar o histórico por rodada e confere a última rodada contra `results/raw/resultados.csv`. Ele **nunca sobrescreve** arquivos existentes: para regenerar, apague antes o conteúdo de `results/export/`.
- Todos os parâmetros do experimento (sementes, α, número de clientes e de rodadas, τ, r, n_piso) ficam em `src/config.py`.

## Estrutura de pastas

```
.
├── data/
│   └── creditcard.csv            # baixar do Kaggle (não versionado)
├── src/
│   ├── config.py                 # parâmetros do experimento
│   ├── data.py                   # split, normalização e partição Dirichlet (Non-IID)
│   ├── balancing.py              # undersampling adaptativo
│   ├── model.py                  # SGDClassifier (regressão logística) e métricas
│   ├── federated.py              # treino local e agregação FedAvg
│   └── run_all.py                # executa todos os experimentos
├── scripts/
│   ├── extrair_resultados.py     # tabelas, diferenças pareadas, ações por cliente, histórico
│   └── gerar_grafico_auprc.py    # figura de AUPRC por rodada
├── results/
│   ├── raw/                      # saída bruta do run_all.py
│   ├── export/                   # tabelas e CSVs gerados pelo extrair_resultados.py
│   └── figures/                  # figuras do artigo
├── requirements.txt
└── README.md
```

## Resultados gerados

| Arquivo | Conteúdo |
|---|---|
| `results/raw/resultados.csv` | Métricas da última rodada de cada execução |
| `results/raw/particoes.csv` | Composição de cada cliente por semente e α |
| `results/export/metricas_finais_por_seed.csv` | Métricas finais por semente |
| `results/export/metricas_resumo.csv` | Média e desvio padrão sobre as sementes (Tabelas II e IV) |
| `results/export/diferencas_pareadas.csv` | Diferenças pareadas por semente (Tabela III) |
| `results/export/acoes_por_cliente.csv`, `acoes_resumo_por_alpha_seed.csv` | Ação do método adaptativo por cliente (Tabela V) |
| `results/export/auprc_por_rodada.csv` | AUPRC ao longo das 20 rodadas |
| `results/export/desvios_ultimas_rodadas.csv` | Desvio padrão da AUPRC nas 5 últimas rodadas (estabilidade) |
| `results/export/config_e_ambiente.txt` | Configuração e versões do ambiente |
| `results/figures/auprc_por_rodada.pdf` | Figura 2 do artigo |

## Citação

Se usar este código, cite o artigo (Lage, A., CEFET-MG, 2026).