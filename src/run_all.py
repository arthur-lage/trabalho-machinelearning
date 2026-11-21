import numpy as np
import pandas as pd
from pathlib import Path

from config import CONFIG
from data import load_and_split, dirichlet_partition
from federated import run_federated   # ajuste para o arquivo onde estão run_federated/fedavg/local_train

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

VARIANTES = {
    "sem_balanceamento": 10**9,
    "adaptativo": CONFIG["tau"],   # tau=10
    "uniforme": 0,
}
PESOS = ["pre", "post"]


def main():
    rows = []        # métricas finais de cada execução
    part_rows = []   # composição de cada cliente (tabela para o artigo)

    for seed in CONFIG["seeds"]:
        X_train, X_test, y_train, y_test = load_and_split(seed=seed)

        for alpha in CONFIG["alphas"]:
            ci = dirichlet_partition(y_train, CONFIG["n_clients"], alpha=alpha, seed=seed)

            # guarda a composição dos clientes (uma vez por seed × alpha)
            y_arr = np.asarray(y_train)
            for k, idx in enumerate(ci):
                n_fraud = int(y_arr[idx].sum())
                part_rows.append({
                    "seed": seed, "alpha": alpha, "cliente": k,
                    "total": len(idx), "fraudes": n_fraud,
                    "legitimas": len(idx) - n_fraud,
                })

            for peso in PESOS:
                for nome, tau in VARIANTES.items():
                    cfg = {**CONFIG, "tau": tau, "weight_by": peso}
                    history = run_federated(X_train, y_train, X_test, y_test, ci, cfg, seed=seed)
                    final = history[-1]
                    rows.append({
                        "seed": seed, "alpha": alpha, "peso": peso,
                        "variante": nome, "tau": tau,
                        **final,
                    })
                    print(f"seed={seed} alpha={alpha} peso={peso:4s} {nome:18s} "
                          f"AUPRC={final['auprc']:.4f} recall={final['recall']:.4f}")

            # salva a cada (seed, alpha), para não perder nada se algo falhar
            pd.DataFrame(rows).to_csv(RESULTS_DIR / "resultados.csv", index=False)
            pd.DataFrame(part_rows).to_csv(RESULTS_DIR / "particoes.csv", index=False)

    # resumo: média ± desvio por combinação
    df = pd.DataFrame(rows)
    resumo = (
        df.groupby(["alpha", "peso", "variante"])[["auprc", "recall", "precision", "f1"]]
          .agg(["mean", "std"])
          .round(4)
    )
    resumo.to_csv(RESULTS_DIR / "resumo.csv")
    print("\n=== RESUMO (média, desvio) ===")
    print(resumo)


if __name__ == "__main__":
    main()