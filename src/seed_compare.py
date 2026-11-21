import pandas as pd

df = pd.read_csv("results/resultados.csv")
p = df.pivot_table(index=["seed", "alpha", "peso"],
                   columns="variante", values="auprc").reset_index()
p["adapt_vs_none"] = p["adaptativo"] - p["sem_balanceamento"]
p["adapt_vs_unif"] = p["adaptativo"] - p["uniforme"]

for col in ["adapt_vs_none", "adapt_vs_unif"]:
    g = p.groupby(["alpha", "peso"])[col]
    out = pd.DataFrame({"media": g.mean(), "std": g.std(),
                        "vitorias_de_5": g.apply(lambda s: (s > 0).sum())})
    print(f"\n{col} (adaptativo − outro)")
    print(out.round(4))