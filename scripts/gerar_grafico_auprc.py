"""Gera a figura de AUPRC por rodada (um painel por alpha), ponderação pré-balanceamento.

Uso (da raiz do projeto):
    python scripts/gerar_grafico_auprc.py

Lê results/export/auprc_por_rodada.csv e grava results/figures/auprc_por_rodada.pdf.
Colunas esperadas no CSV: alpha, metodo, variante_peso, seed, rodada, auprc
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "export" / "auprc_por_rodada.csv"
FIG_DIR = BASE_DIR / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SAIDA = FIG_DIR / "auprc_por_rodada.pdf"

ALPHAS = [0.1, 0.5, 1.0]
METODOS = {  # nome no CSV: (rótulo, cor, estilo de linha)
    "sem_balanceamento": ("Sem balanceamento", "#444444", "-"),
    "random_undersampling_tau0": ("Sem suspensão (τ=0)", "#d95f02", "--"),
    "adaptativo": ("Adaptativo", "#1b6ca8", "-"),
}

df = pd.read_csv(CSV_PATH)
df = df[df["variante_peso"] == "pre"]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)

for ax, alpha in zip(axes, ALPHAS):
    for metodo, (rotulo, cor, estilo) in METODOS.items():
        g = df[(df["alpha"] == alpha) & (df["metodo"] == metodo)]
        # linhas = rodadas, colunas = sementes
        piv = g.pivot(index="rodada", columns="seed", values="auprc")
        x = piv.index + 1  # rodadas numeradas a partir de 1
        media = piv.mean(axis=1)
        dp = piv.std(axis=1)  # desvio padrão amostral entre sementes
        ax.plot(x, media, color=cor, ls=estilo, lw=1.4, label=rotulo)
        ax.fill_between(x, (media - dp).clip(lower=0), (media + dp).clip(upper=1),
                        color=cor, alpha=0.12, lw=0)
    ax.set_title("α = " + str(alpha).replace(".", ","), fontsize=9)
    ax.set_xlabel("Rodada", fontsize=8)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("AUPRC", fontsize=8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(SAIDA, bbox_inches="tight")
print(f"Figura salva em {SAIDA}")