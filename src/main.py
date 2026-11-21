from config import CONFIG
from model import evaluate, make_model
from balancing import adaptive_undersample
from data import load_and_split, dirichlet_partition
from federated import local_train, fedavg, run_federated

import numpy as np

X_train, X_test, y_train, y_test = load_and_split(seed=0)

if __name__ == "__main__":
    # X_train, X_test, y_train, y_test = load_and_split(seed=0)
    # ci = dirichlet_partition(y_train, CONFIG["n_clients"], alpha=0.5, seed=0)

    # for k, idx in enumerate(ci):
    #     Xk, yk = X_train.iloc[idx], y_train.iloc[idx]
    #     Xb, yb = adaptive_undersample(Xk, yk, CONFIG["tau"], CONFIG["r"],
    #                                   CONFIG["n_piso"], seed=k)
    #     print(f"cliente {k}: antes={len(yk):6d} (fraudes={int(yk.sum())})  "
    #           f"depois={len(yb):6d} (fraudes={int(yb.sum())})")

    # X_train, X_test, y_train, y_test = load_and_split(seed=0)
    # model = make_model(seed = 0)
    # model.fit(X_train, y_train)
    # print(evaluate(model, X_test, y_test))

    # X_train, X_test, y_train, y_test = load_and_split(seed=0)
    # ci = dirichlet_partition(y_train, CONFIG["n_clients"], alpha=0.5, seed=0)

    # Xk, yk = X_train.iloc[ci[3]], y_train.iloc[ci[3]]

    # zeros = [np.zeros((1, 30)), np.zeros(1)]
    # params, n = local_train(zeros, Xk, yk, CONFIG, seed=3)
    # print(n, np.abs(params[0]).sum())   # n=4081, soma > 0
    # print(np.abs(zeros[0]).sum())       # tem que continuar 0.0

    # a = ([np.ones((1, 30)), np.ones(1)], 100)
    # b = ([np.full((1, 30), 3.0), np.full(1, 3.0)], 300)
    # out = fedavg([a, b])
    # print(out[0][0, :3], out[1])   # esperado: [2.5 2.5 2.5] [2.5]
    # print(out[0].shape, out[1].shape)

    # X_train, X_test, y_train, y_test = load_and_split(seed=0)
    # # ci = dirichlet_partition(y_train, CONFIG["n_clients"], alpha=0.5, seed=0)
    # ci = dirichlet_partition(y_train, CONFIG["n_clients"], alpha=100, seed=0)
    # history = run_federated(X_train, y_train, X_test, y_test, ci, CONFIG, seed=0)
    # history = run_federated(X_train, y_train, X_test, y_test, ci, CONFIG, seed=0)
    # history = run_federated(X_train, y_train, X_test, y_test, ci, CONFIG, seed=0)

    # for r, m in enumerate(history):
    #     if r % 5 == 0 or r == len(history) - 1:
    #         print(r, m)

    # X_train, X_test, y_train, y_test = load_and_split(seed=0)
    # ci = dirichlet_partition(y_train, CONFIG["n_clients"], alpha=0.5, seed=0)

    # variantes = {
    #     "sem balanceamento (tau=1e9)": 10**9,
    #     "adaptativo (tau=10)": 10,
    #     "uniforme (tau=0)": 0,
    # }

    # for nome, tau in variantes.items():
    #     cfg = {**CONFIG, "tau": tau}   # cópia: não altera o CONFIG original
    #     history = run_federated(X_train, y_train, X_test, y_test, ci, cfg, seed=0)
    #     final = history[-1]
    #     print(f"{nome:30s} AUPRC={final['auprc']:.4f}  "
    #           f"recall={final['recall']:.4f}  precision={final['precision']:.4f}  "
    #           f"f1={final['f1']:.4f}")
    X_train, X_test, y_train, y_test = load_and_split(seed=0)
    ci = dirichlet_partition(y_train, CONFIG["n_clients"], alpha=0.5, seed=0)

    variantes = {
        "sem balanceamento": 10**9,
        "adaptativo (tau=10)": 10,
        "uniforme (tau=0)": 0,
    }

    for weight_by in ["post", "pre"]:
        print(f"\n--- peso = {weight_by} ---")
        for nome, tau in variantes.items():
            cfg = {**CONFIG, "tau": tau, "weight_by": weight_by}
            history = run_federated(X_train, y_train, X_test, y_test, ci, cfg, seed=0)
            f = history[-1]
            print(f"{nome:22s} AUPRC={f['auprc']:.4f}  recall={f['recall']:.4f}  "
                  f"precision={f['precision']:.4f}")