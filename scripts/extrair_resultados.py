#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extrai os resultados dos experimentos para a seção de Resultados do artigo.

Este script NÃO altera a lógica dos experimentos, hiperparâmetros, seeds ou split.
Ele lê os arquivos já existentes em results/raw/ e, quando necessário, reexecuta o
código atual apenas para registrar o histórico por rodada (que não era salvo).

Uso:
    python scripts/extrair_resultados.py
    python scripts/extrair_resultados.py --sem-historico

Saídas em results/export/ (arquivos existentes nunca são sobrescritos):
    1. metricas_finais_por_seed.csv
    2. metricas_resumo.csv
    3. diferencas_pareadas.csv
    4. acoes_por_cliente.csv
    4b. acoes_resumo_por_alpha_seed.csv  (resumo compacto do item 4)
    5. auprc_por_rodada.csv
    5b. desvios_ultimas_rodadas.csv  (desvio da AUPRC nas 5 últimas rodadas)
    6. config_e_ambiente.txt
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "raw"
EXPORT_DIR = BASE_DIR / "results" / "export"
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import CONFIG  # noqa: E402

# Mapeamento entre o nome usado em results/raw/resultados.csv e o nome de exibição.
METODO_POR_VARIANTE = {
    "sem_balanceamento": "sem_balanceamento",
    "adaptativo": "adaptativo",
    "uniforme": "random_undersampling_tau0",
}

# Exatamente as variantes usadas em src/run_all.py.
VARIANTES = {
    "sem_balanceamento": 10**9,
    "adaptativo": CONFIG["tau"],  # 10
    "uniforme": 0,
}

PESOS = ["pre", "post"]
METRICAS = ["precision", "recall", "f1", "auprc"]
METRICAS_RESUMO = ["precision", "recall", "f1", "auprc"]

# Número de rodadas finais usadas no desvio de estabilidade (item 5b).
N_ULTIMAS_RODADAS = 5


# ---------------------------------------------------------------------------
# Utilidades de escrita (nunca sobrescrever)
# ---------------------------------------------------------------------------
def escrever_csv(df: pd.DataFrame, path: Path, **kwargs) -> bool:
    if path.exists():
        print(f"[SKIP] {path} já existe; não foi sobrescrito.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", **kwargs)
    print(f"[OK]   {path}  ({len(df)} linhas)")
    return True


def escrever_texto(texto: str, path: Path) -> bool:
    if path.exists():
        print(f"[SKIP] {path} já existe; não foi sobrescrito.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(texto, encoding="utf-8")
    print(f"[OK]   {path}")
    return True


# ---------------------------------------------------------------------------
# Leitura e validação dos resultados brutos
# ---------------------------------------------------------------------------
def carregar_resultados() -> pd.DataFrame:
    path = RESULTS_DIR / "resultados.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode primeiro: python src/run_all.py"
        )

    df = pd.read_csv(path)
    obrigatorias = {
        "seed", "alpha", "peso", "variante", "tau",
        *METRICAS,
    }
    faltando = obrigatorias - set(df.columns)
    if faltando:
        raise ValueError(f"results/raw/resultados.csv sem colunas: {sorted(faltando)}")

    df["metodo"] = df["variante"].map(METODO_POR_VARIANTE)
    if df["metodo"].isna().any():
        desconhecidas = sorted(df.loc[df["metodo"].isna(), "variante"].unique())
        raise ValueError(f"Variantes sem mapeamento: {desconhecidas}")

    df = df.rename(columns={"peso": "variante_peso"})
    df = df.sort_values(["alpha", "metodo", "variante_peso", "seed"]).reset_index(drop=True)
    return df


def validar_completude(df: pd.DataFrame) -> list[str]:
    problemas: list[str] = []

    esperado = pd.MultiIndex.from_product(
        [CONFIG["seeds"], CONFIG["alphas"], PESOS, list(VARIANTES)],
        names=["seed", "alpha", "variante_peso", "variante"],
    ).to_frame(index=False)
    esperado["metodo"] = esperado["variante"].map(METODO_POR_VARIANTE)
    esperado = esperado[["seed", "alpha", "variante_peso", "metodo"]]

    chave = ["seed", "alpha", "variante_peso", "metodo"]
    atual = df[chave].drop_duplicates()
    faltantes = esperado.merge(atual, on=chave, how="left", indicator=True)
    faltantes = faltantes[faltantes["_merge"] == "left_only"]
    if len(faltantes):
        problemas.append(
            f"{len(faltantes)} combinações esperadas não estão em resultados.csv: "
            + "; ".join(
                f"seed={int(r.seed)}, alpha={r.alpha}, peso={r.variante_peso}, metodo={r.metodo}"
                for r in faltantes.itertuples()
            )
        )

    duplicadas = df.duplicated(subset=chave, keep=False)
    if duplicadas.any():
        problemas.append(
            f"{int(duplicadas.sum())} linhas duplicadas para a mesma combinação "
            "seed/alpha/peso/metodo em resultados.csv."
        )

    for col in METRICAS:
        n_nan = int(df[col].isna().sum())
        if n_nan:
            problemas.append(f"{n_nan} valores NaN em {col}.")
        fora = df[(df[col] < 0) | (df[col] > 1)]
        if len(fora):
            problemas.append(f"{len(fora)} valores fora de [0,1] em {col}.")

    return problemas


# ---------------------------------------------------------------------------
# 1. Métricas finais por seed
# ---------------------------------------------------------------------------
def gerar_metricas_finais(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        ["alpha", "metodo", "variante_peso", "seed", *METRICAS]
    ].copy()
    out = out.sort_values(["alpha", "metodo", "variante_peso", "seed"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2. Resumo (média e desvio padrão amostral sobre as seeds)
# ---------------------------------------------------------------------------
def gerar_resumo(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["alpha", "metodo", "variante_peso"], sort=True)
    out = grouped.agg(
        precision_media=("precision", "mean"),
        precision_dp=("precision", "std"),
        recall_media=("recall", "mean"),
        recall_dp=("recall", "std"),
        f1_media=("f1", "mean"),
        f1_dp=("f1", "std"),
        auprc_media=("auprc", "mean"),
        auprc_dp=("auprc", "std"),
    ).reset_index()
    return out


# ---------------------------------------------------------------------------
# 3. Diferenças pareadas (adaptativo - cada baseline), por seed
# ---------------------------------------------------------------------------
def gerar_diferencas(df: pd.DataFrame) -> pd.DataFrame:
    linhas: list[dict] = []
    baselines = ["sem_balanceamento", "random_undersampling_tau0"]

    for alpha in sorted(df["alpha"].unique()):
        for peso in PESOS:
            sub = df[(df["alpha"] == alpha) & (df["variante_peso"] == peso)]
            adapt = sub[sub["metodo"] == "adaptativo"].set_index("seed")
            if adapt.empty:
                continue

            for baseline in baselines:
                base = sub[sub["metodo"] == baseline].set_index("seed")
                if base.empty:
                    continue

                for metrica in ["auprc", "f1"]:
                    dif = (adapt[metrica] - base[metrica]).dropna()
                    if dif.empty:
                        continue

                    for seed, valor in dif.items():
                        linhas.append({
                            "alpha": alpha,
                            "variante_peso": peso,
                            "baseline": baseline,
                            "metrica": metrica,
                            "tipo": "seed",
                            "seed": int(seed),
                            "diferenca": float(valor),
                            "media_diferenca": np.nan,
                            "desvio_diferenca": np.nan,
                            "n_seeds_adaptativo_melhor": np.nan,
                            "n_seeds": np.nan,
                            "n_seeds_empate": np.nan,
                        })

                    linhas.append({
                        "alpha": alpha,
                        "variante_peso": peso,
                        "baseline": baseline,
                        "metrica": metrica,
                        "tipo": "resumo",
                        "seed": "",
                        "diferenca": np.nan,
                        "media_diferenca": float(dif.mean()),
                        "desvio_diferenca": float(dif.std(ddof=1)),
                        "n_seeds_adaptativo_melhor": int((dif > 0).sum()),
                        "n_seeds": int(dif.notna().sum()),
                        "n_seeds_empate": int((dif == 0).sum()),
                    })

    cols = [
        "alpha", "variante_peso", "baseline", "metrica", "tipo", "seed",
        "diferenca", "media_diferenca", "desvio_diferenca",
        "n_seeds_adaptativo_melhor", "n_seeds", "n_seeds_empate",
    ]
    return pd.DataFrame(linhas, columns=cols)


# ---------------------------------------------------------------------------
# 4. Ações do método adaptativo por cliente
# ---------------------------------------------------------------------------
def gerar_acoes() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = RESULTS_DIR / "particoes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} não encontrado. Rode primeiro: python src/run_all.py"
        )

    part = pd.read_csv(path)
    obrigatorias = {"seed", "alpha", "cliente", "total", "fraudes", "legitimas"}
    faltando = obrigatorias - set(part.columns)
    if faltando:
        raise ValueError(f"results/raw/particoes.csv sem colunas: {sorted(faltando)}")

    tau = CONFIG["tau"]
    r = CONFIG["r"]
    n_piso = CONFIG["n_piso"]

    linhas: list[dict] = []
    for row in part.itertuples(index=False):
        n_fraudes = int(row.fraudes)
        n_legit = int(row.legitimas)

        if n_fraudes < tau:
            # adaptive_undersample devolve X, y intactos: o undersampling é
            # suspenso, mas o cliente CONTINUA treinando com todos os dados.
            acao = "suspender"
            n_legit_usadas = n_legit
        else:
            n_keep = max(r * n_fraudes, n_piso)
            if n_legit <= n_keep:
                acao = "manter"
                n_legit_usadas = n_legit
            else:
                acao = "undersampling"
                n_legit_usadas = n_keep

        linhas.append({
            "alpha": float(row.alpha),
            "seed": int(row.seed),
            "cliente": int(row.cliente),
            "n_fraudes": n_fraudes,
            "n_legitimas_originais": n_legit,
            "acao": acao,
            "n_legitimas_usadas": int(n_legit_usadas),
            "n_total_usado_no_treino": int(n_fraudes + n_legit_usadas),
        })

    acoes = pd.DataFrame(linhas)

    resumo = acoes.groupby(["alpha", "seed"], sort=True).agg(
        n_clientes=("cliente", "count"),
        total_clientes_suspensos=("acao", lambda s: int((s == "suspender").sum())),
        n_clientes_undersampling=("acao", lambda s: int((s == "undersampling").sum())),
        n_clientes_manter=("acao", lambda s: int((s == "manter").sum())),
        fraudes_total=("n_fraudes", "sum"),
        legitimas_originais_total=("n_legitimas_originais", "sum"),
        legitimas_usadas_total=("n_legitimas_usadas", "sum"),
    ).reset_index()
    resumo["pct_legitimas_mantidas_no_treino"] = (
        100.0 * resumo["legitimas_usadas_total"] / resumo["legitimas_originais_total"]
    )

    acoes = acoes.merge(
        resumo[
            [
                "alpha", "seed", "total_clientes_suspensos",
                "pct_legitimas_mantidas_no_treino",
            ]
        ],
        on=["alpha", "seed"],
        how="left",
    )
    acoes = acoes[
        [
            "alpha", "seed", "cliente", "n_fraudes", "n_legitimas_originais",
            "acao", "n_legitimas_usadas", "n_total_usado_no_treino",
            "total_clientes_suspensos", "pct_legitimas_mantidas_no_treino",
        ]
    ].sort_values(["alpha", "seed", "cliente"]).reset_index(drop=True)

    resumo = resumo[
        [
            "alpha", "seed", "n_clientes", "total_clientes_suspensos",
            "n_clientes_undersampling", "n_clientes_manter", "fraudes_total",
            "legitimas_originais_total", "legitimas_usadas_total",
            "pct_legitimas_mantidas_no_treino",
        ]
    ].sort_values(["alpha", "seed"]).reset_index(drop=True)

    return acoes, resumo


# ---------------------------------------------------------------------------
# 5. Histórico por rodada (reexecuta o código atual, sem alterar a lógica)
# ---------------------------------------------------------------------------
def gerar_historico() -> pd.DataFrame:
    from data import load_and_split, dirichlet_partition
    from federated import run_federated

    linhas: list[dict] = []
    total = len(CONFIG["seeds"]) * len(CONFIG["alphas"])
    feitos = 0

    for seed in CONFIG["seeds"]:
        X_train, X_test, y_train, y_test = load_and_split(seed=seed)
        for alpha in CONFIG["alphas"]:
            ci = dirichlet_partition(
                y_train, CONFIG["n_clients"], alpha=alpha, seed=seed
            )
            for peso in PESOS:
                for variante, tau in VARIANTES.items():
                    cfg = {**CONFIG, "tau": tau, "weight_by": peso}
                    history = run_federated(
                        X_train, y_train, X_test, y_test, ci, cfg, seed=seed
                    )
                    for rodada, metricas in enumerate(history):
                        linhas.append({
                            "alpha": alpha,
                            "metodo": METODO_POR_VARIANTE[variante],
                            "variante_peso": peso,
                            "seed": seed,
                            "rodada": rodada,
                            "auprc": metricas["auprc"],
                            "f1": metricas["f1"],
                            "recall": metricas["recall"],
                            "precision": metricas["precision"],
                        })
            feitos += 1
            print(f"  histórico: {feitos}/{total} (seed={seed}, alpha={alpha})")

    out = pd.DataFrame(linhas)
    out = out.sort_values(
        ["alpha", "metodo", "variante_peso", "seed", "rodada"]
    ).reset_index(drop=True)
    return out


def validar_historico(hist: pd.DataFrame, finais: pd.DataFrame) -> tuple[float, list[str]]:
    chave = ["alpha", "metodo", "variante_peso", "seed"]
    ultima = hist[hist["rodada"] == hist.groupby(chave)["rodada"].transform("max")]
    merged = ultima.merge(finais, on=chave, suffixes=("_hist", "_csv"), how="outer")

    problemas: list[str] = []

    if len(merged) != len(finais):
        problemas.append(
            f"histórico tem {len(merged)} grupos na última rodada; "
            f"resultados.csv tem {len(finais)}."
        )

    max_diff = 0.0
    for metrica in METRICAS:
        a = merged[f"{metrica}_hist"]
        b = merged[f"{metrica}_csv"]
        diff = (a - b).abs()
        if diff.notna().any():
            max_diff = max(max_diff, float(diff.max()))
        ruins = merged[diff > 1e-8]
        if len(ruins):
            problemas.append(
                f"{len(ruins)} grupos com {metrica} da última rodada diferente "
                f"de resultados.csv (diferença máxima {float(diff.max()):.3e})."
            )

    return max_diff, problemas


# ---------------------------------------------------------------------------
# 5b. Desvio da AUPRC nas últimas rodadas (estabilidade)
# ---------------------------------------------------------------------------
def gerar_desvios_ultimas_rodadas(
    hist: pd.DataFrame, n: int = N_ULTIMAS_RODADAS
) -> pd.DataFrame:
    """Desvio padrão amostral (ddof=1) da AUPRC nas n últimas rodadas.

    Calculado entre as n rodadas dentro de cada semente e depois com média
    entre as sementes. NÃO é o desvio entre sementes das tabelas de resultados.
    Gera uma linha por alpha x metodo x variante_peso.
    """
    chave = ["alpha", "metodo", "variante_peso"]
    rodada_max = int(hist["rodada"].max())
    ultimas = hist[hist["rodada"] > rodada_max - n]

    contagem = ultimas.groupby(chave + ["seed"])["rodada"].nunique()
    if (contagem != n).any():
        raise ValueError(f"Algum grupo não tem exatamente {n} rodadas finais.")

    por_semente = ultimas.groupby(chave + ["seed"])["auprc"].std(ddof=1)
    out = (
        por_semente.groupby(chave)
        .agg(desvio_auprc_ultimas_rodadas="mean", n_seeds="count")
        .reset_index()
    )
    out["rodadas_usadas"] = f"{int(ultimas['rodada'].min())}-{rodada_max}"
    return out.sort_values(chave).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Configuração e ambiente
# ---------------------------------------------------------------------------
def gerar_texto_config() -> str:
    import sklearn

    from data import load_and_split

    df = pd.read_csv(BASE_DIR / "data" / "creditcard.csv")
    n_linhas, n_colunas = df.shape
    n_fraudes = int(df["Class"].sum())

    linhas_teste = []
    for seed in CONFIG["seeds"]:
        _, X_test, _, y_test = load_and_split(seed=seed)
        linhas_teste.append(
            f"  seed={seed}: teste={len(y_test)} transações, "
            f"{int(y_test.sum())} fraudes; treino={len(df) - len(y_test)} "
            f"transações, {n_fraudes - int(y_test.sum())} fraudes"
        )

    texto = f"""CONFIGURAÇÃO E AMBIENTE DOS EXPERIMENTOS
========================================
Gerado por scripts/extrair_resultados.py

Ambiente
--------
Python: {platform.python_version()} ({platform.platform()})
scikit-learn: {sklearn.__version__}
NumPy: {np.__version__}
pandas: {pd.__version__}

Dados
-----
Arquivo: data/creditcard.csv
Dimensões: {n_linhas} linhas x {n_colunas} colunas
Fraudes no dataset completo: {n_fraudes}

Split (src/data.py)
-------------------
train_test_split(test_size={CONFIG['test_size']}, stratify=y, random_state=seed)
Por seed:
{chr(10).join(linhas_teste)}

Hiperparâmetros (src/config.py e src/model.py)
----------------------------------------------
seeds = {CONFIG['seeds']}
alphas = {CONFIG['alphas']}
K = n_clients = {CONFIG['n_clients']} (partição Dirichlet, min_size=10, max_tries=1000)
rounds = {CONFIG['rounds']}
local_epochs = {CONFIG['local_epochs']}
tau = {CONFIG['tau']} (método adaptativo)
r = {CONFIG['r']}
n_piso = {CONFIG['n_piso']}
lambda = 1e-4 (parâmetro alpha do SGDClassifier, penalty="l2")
eta = 0.01 (eta0, learning_rate="constant")
Modelo = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                       learning_rate="constant", eta0=0.01, random_state=seed)
Limiar de decisão = 0.5 (model.predict padrão)
Métrica de ranking = average_precision_score (AUPRC)

Métodos de balanceamento implementados
--------------------------------------
1) sem_balanceamento: tau = 1e9. Como n_fraudes < tau sempre, adaptive_undersample
   devolve X e y intactos.
2) adaptativo: tau = 10, r = 10, n_piso = 100.
   - se n_fraudes < 10: devolve X e y intactos (undersampling suspenso; o cliente
     CONTINUA treinando com todos os dados);
   - senão: n_keep = max(10 * n_fraudes, 100) legítimas; se houver mais legítimas
     que n_keep, sorteia n_keep sem reposição; todas as fraudes são mantidas.
3) random_undersampling_tau0 (chamado de "uniforme" em results/raw/): tau = 0,
   r = 10, n_piso = 100.
   - como n_fraudes < 0 nunca ocorre, a regra de suspensão não é acionada;
   - n_keep = max(10 * n_fraudes, 100). Portanto NÃO é um undersampling uniforme
     para 100 exemplos: é a mesma regra do adaptativo, mas aplicada também aos
     clientes com 0 a 9 fraudes.
   - para clientes com >= 10 fraudes, adaptativo e tau=0 são idênticos.

SMOTE: NÃO IMPLEMENTADO no código atual. Não há valor de k vizinhos a reportar
       e não há resultados salvos para esse método.
Class Weight: NÃO IMPLEMENTADO no código atual. Não há valor de class_weight a
       reportar e não há resultados salvos para esse método.

Variantes de peso do FedAvg (src/federated.py)
----------------------------------------------
pre:  n_examples = len(y) original do cliente (antes do balanceamento)
post: n_examples = len(y) após o undersampling
O FedAvg usado é a média ponderada de coef_ e intercept_ pelo n_examples.

Arquivos de origem
------------------
results/raw/resultados.csv: métricas da última rodada de cada execução
results/raw/particoes.csv: composição de cada cliente por seed x alpha
results/raw/resumo.csv: resumo antigo (arredondado) gerado por src/run_all.py

Observações
-----------
- As métricas de metricas_finais_por_seed.csv vêm de results/raw/resultados.csv
  (última rodada, teste centralizado, limiar 0,5).
- O desvio padrão em metricas_resumo.csv é o desvio padrão amostral (ddof=1)
  sobre as {len(CONFIG['seeds'])} seeds.
- O histórico por rodada não estava salvo em results/raw. O arquivo
  auprc_por_rodada.csv foi gerado reexecutando o código atual com as mesmas
  seeds, alphas, hiperparâmetros e split. A última rodada foi conferida contra
  results/raw/resultados.csv. A coluna "rodada" começa em 0 (rodadas 0 a
  {CONFIG['rounds'] - 1}); a Figura 2 do artigo mostra 1 a {CONFIG['rounds']}.
- desvios_ultimas_rodadas.csv: desvio padrão amostral (ddof=1) da AUPRC nas
  {N_ULTIMAS_RODADAS} últimas rodadas, calculado entre as rodadas dentro de cada
  seed e depois com média entre as seeds (não é o desvio entre seeds).
- As ações por cliente foram reconstruídas a partir de results/raw/particoes.csv e
  da regra determinística de adaptive_undersample. O código não registra os
  índices exatos sorteados em cada rodada, apenas a quantidade de legítimas
  mantidas.
- "suspender" significa que o undersampling foi suspenso para aquele cliente;
  ele NÃO foi removido do treinamento federado.

Comandos
--------
# Regenerar os resultados brutos (não executado por este script):
python src/run_all.py

# Gerar a exportação:
python scripts/extrair_resultados.py

# Gerar a exportação sem reexecutar o histórico por rodada:
python scripts/extrair_resultados.py --sem-historico
"""
    return texto


# ---------------------------------------------------------------------------
# Impressão final
# ---------------------------------------------------------------------------
def imprimir_tabela(resumo: pd.DataFrame) -> None:
    print("\n## AUPRC e F1 (média ± dp sobre as seeds)\n")
    print("| alpha | método | variante_peso | AUPRC (média ± dp) | F1 (média ± dp) |")
    print("|---:|---|---|---:|---:|")
    for row in resumo.sort_values(["alpha", "metodo", "variante_peso"]).itertuples():
        auprc = f"{row.auprc_media:.4f} ± {row.auprc_dp:.4f}"
        f1 = f"{row.f1_media:.4f} ± {row.f1_dp:.4f}"
        print(f"| {row.alpha} | {row.metodo} | {row.variante_peso} | {auprc} | {f1} |")


def imprimir_desvios(desvios: pd.DataFrame) -> None:
    print(f"\n## Desvio da AUPRC nas {N_ULTIMAS_RODADAS} últimas rodadas "
          "(ddof=1, dentro da seed; média entre seeds)\n")
    print("| alpha | método | variante_peso | desvio |")
    print("|---:|---|---|---:|")
    for row in desvios.itertuples():
        print(f"| {row.alpha} | {row.metodo} | {row.variante_peso} | "
              f"{row.desvio_auprc_ultimas_rodadas:.3f} |")


def imprimir_ausentes(historico_ok: bool, problemas_historico: list[str]) -> None:
    print("\n## Dados pedidos que não foi possível gerar / não existem\n")
    print("- **SMOTE**: não implementado no código atual. Não existe resultado "
          "salvo, nem valor de k vizinhos a reportar.")
    print("- **Class Weight**: não implementado no código atual. Não existe "
          "resultado salvo, nem valor de class_weight a reportar.")
    if historico_ok:
        print("- **Histórico por rodada**: não estava salvo em `results/`; foi "
              "regenerado executando o código atual e conferido contra "
              "`results/raw/resultados.csv`.")
    else:
        print("- **Histórico por rodada**: não estava salvo em `results/` e a "
              "regeneração falhou. Rode `python scripts/extrair_resultados.py` "
              "novamente após corrigir o erro.")
    print("- **Índices exatos das legítimas sorteadas em cada rodada**: não são "
          "registrados pelo código. `acoes_por_cliente.csv` contém as quantidades "
          "determinísticas (n_legitimas_usadas), não a amostra específica.")
    print("- **Métricas por cliente**: o código atual não as salva.")
    if problemas_historico:
        print("- Problemas na validação do histórico:")
        for p in problemas_historico:
            print(f"  - {p}")


def imprimir_anomalias(
    df: pd.DataFrame,
    resumo: pd.DataFrame,
    diferencas: pd.DataFrame,
    problemas_completude: list[str],
    max_diff_hist: float | None,
    problemas_hist: list[str],
) -> None:
    print("\n## Alertas e resultados que contrariam o esperado\n")
    achou = False

    for p in problemas_completude:
        print(f"- {p}")
        achou = True

    # F1 zero / precision zero
    zero_f1 = df[df["f1"] == 0]
    if len(zero_f1):
        print(f"- **{len(zero_f1)} execuções com F1 = 0** (o limiar 0,5 não "
              "produziu nenhum positivo previsto), apesar de AUPRC > 0 em várias "
              "delas:")
        for r in zero_f1.sort_values(["alpha", "metodo", "variante_peso", "seed"]).itertuples():
            print(f"  - seed={r.seed}, alpha={r.alpha}, {r.metodo}, "
                  f"{r.variante_peso}: AUPRC={r.auprc:.4f}")
        achou = True

    # AUPRC muito baixa (próxima do aleatório)
    quase_aleatorio = df[df["auprc"] < 0.01]
    if len(quase_aleatorio):
        print(f"- **{len(quase_aleatorio)} execuções com AUPRC < 0,01** "
              "(praticamente aleatório para esta prevalência):")
        for r in quase_aleatorio.itertuples():
            print(f"  - seed={r.seed}, alpha={r.alpha}, {r.metodo}, "
                  f"{r.variante_peso}: AUPRC={r.auprc:.6f}, F1={r.f1:.4f}")
        achou = True

    # Adaptativo pior que sem balanceamento (média)
    print("- Comparação pareada adaptativo − sem_balanceamento (média sobre seeds):")
    for row in diferencas[
        (diferencas["tipo"] == "resumo")
        & (diferencas["baseline"] == "sem_balanceamento")
    ].sort_values(["alpha", "variante_peso", "metrica"]).itertuples():
        sinal = "PIOR" if row.media_diferenca < 0 else "melhor"
        print(f"  - alpha={row.alpha}, peso={row.variante_peso}, {row.metrica}: "
              f"{row.media_diferenca:+.4f} ({sinal}); "
              f"adaptativo melhor em {int(row.n_seeds_adaptativo_melhor)}/"
              f"{int(row.n_seeds)} seeds, empates={int(row.n_seeds_empate)}")
        if row.media_diferenca < 0:
            achou = True

    # Dispersão alta entre seeds
    print("- Grupos com dispersão alta entre seeds (AUPRC max − min > 0,20):")
    dispersao = (
        df.groupby(["alpha", "metodo", "variante_peso"])["auprc"]
        .agg(lambda s: float(s.max() - s.min()))
        .reset_index(name="amplitude")
    )
    altos = dispersao[dispersao["amplitude"] > 0.20]
    if len(altos):
        for r in altos.sort_values("amplitude", ascending=False).itertuples():
            print(f"  - alpha={r.alpha}, {r.metodo}, {r.variante_peso}: "
                  f"amplitude AUPRC={r.amplitude:.4f}")
        achou = True
    else:
        print("  - nenhum.")

    # pre == post
    piv = df.pivot_table(
        index=["alpha", "metodo", "seed"],
        columns="variante_peso",
        values="auprc",
        aggfunc="first",
    ).reset_index()
    iguais = piv[np.isclose(piv["pre"], piv["post"], rtol=0, atol=1e-12)]
    if len(iguais):
        print(f"- **{len(iguais)} grupos seed/alpha/método com AUPRC(pre) = "
              "AUPRC(post)**. Isso é esperado quando o undersampling não altera "
              "o número de exemplos de nenhum cliente (ou quando todos os "
              "clientes têm o mesmo tamanho efetivo); não é, por si só, erro.")
        achou = True

    # Histórico
    if max_diff_hist is not None:
        print(f"- Validação do histórico: diferença máxima absoluta entre a "
              f"última rodada reexecutada e results/raw/resultados.csv = "
              f"{max_diff_hist:.3e}.")
        if max_diff_hist > 1e-8:
            print("  - ATENÇÃO: a reexecução não reproduziu exatamente os "
                  "resultados salvos. Verifique ambiente/versões.")
            achou = True
    for p in problemas_hist:
        print(f"- {p}")
        achou = True

    if not achou:
        print("- Nenhum alerta automático.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sem-historico",
        action="store_true",
        help="não reexecuta os experimentos para gerar auprc_por_rodada.csv",
    )
    args = parser.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Lendo results/raw/resultados.csv ...")
    df = carregar_resultados()
    print(f"  {len(df)} execuções carregadas.")

    problemas_completude = validar_completude(df)

    finais = gerar_metricas_finais(df)
    resumo = gerar_resumo(df)
    diferencas = gerar_diferencas(df)

    escrever_csv(finais, EXPORT_DIR / "metricas_finais_por_seed.csv")
    escrever_csv(resumo, EXPORT_DIR / "metricas_resumo.csv")
    escrever_csv(diferencas, EXPORT_DIR / "diferencas_pareadas.csv")

    print("Reconstruindo ações por cliente a partir de results/raw/particoes.csv ...")
    acoes, acoes_resumo = gerar_acoes()
    escrever_csv(acoes, EXPORT_DIR / "acoes_por_cliente.csv")
    escrever_csv(acoes_resumo, EXPORT_DIR / "acoes_resumo_por_alpha_seed.csv")

    historico_ok = False
    problemas_hist: list[str] = []
    max_diff_hist: float | None = None
    hist_path = EXPORT_DIR / "auprc_por_rodada.csv"
    hist: pd.DataFrame | None = None

    if args.sem_historico:
        print("[SKIP] Histórico por rodada desativado por --sem-historico.")
        if hist_path.exists():
            hist = pd.read_csv(hist_path)
    elif hist_path.exists():
        print(f"[SKIP] {hist_path} já existe; lendo para validar.")
        try:
            hist = pd.read_csv(hist_path)
            max_diff_hist, problemas_hist = validar_historico(hist, finais)
            historico_ok = True
        except Exception as exc:  # noqa: BLE001
            problemas_hist.append(f"Falha ao validar histórico existente: {exc}")
    else:
        print("Gerando histórico por rodada (reexecutando o código atual) ...")
        try:
            hist = gerar_historico()
            escrever_csv(hist, hist_path)
            max_diff_hist, problemas_hist = validar_historico(hist, finais)
            historico_ok = True
        except Exception as exc:  # noqa: BLE001
            problemas_hist.append(f"Falha ao gerar histórico: {exc}")
            print(f"[ERRO] {exc}")

    desvios: pd.DataFrame | None = None
    if hist is not None:
        print(f"Calculando desvios da AUPRC nas {N_ULTIMAS_RODADAS} últimas rodadas ...")
        try:
            desvios = gerar_desvios_ultimas_rodadas(hist, n=N_ULTIMAS_RODADAS)
            escrever_csv(
                desvios,
                EXPORT_DIR / "desvios_ultimas_rodadas.csv",
                float_format="%.6f",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERRO] Falha ao gerar desvios_ultimas_rodadas.csv: {exc}")
    else:
        print("[SKIP] Sem histórico disponível; desvios_ultimas_rodadas.csv não gerado.")

    print("Gerando config_e_ambiente.txt ...")
    try:
        texto = gerar_texto_config()
        escrever_texto(texto, EXPORT_DIR / "config_e_ambiente.txt")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERRO] Falha ao gerar config_e_ambiente.txt: {exc}")

    imprimir_tabela(resumo)
    if desvios is not None:
        imprimir_desvios(desvios)
    imprimir_ausentes(historico_ok, problemas_hist)
    imprimir_anomalias(
        df, resumo, diferencas, problemas_completude, max_diff_hist, problemas_hist
    )

    print(f"\nArquivos em: {EXPORT_DIR}")


if __name__ == "__main__":
    main()