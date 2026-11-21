from model import make_model, get_params, set_params, evaluate
from balancing import adaptive_undersample

import numpy as np

# n_exampels = tamanho depois do undersampling

def local_train(global_params, X, y, config, seed):
    model = make_model(seed)
    set_params(model, global_params)

    new_X, new_y = adaptive_undersample(X, y, config["tau"], config["r"], config["n_piso"], seed)

    for _ in range(config["local_epochs"]):
        model.partial_fit(new_X.to_numpy(), new_y.to_numpy(), classes=np.array([0, 1]))

    n_examples = len(new_y) if config["weight_by"] == "post" else len(y)

    return (get_params(model), n_examples)

def fedavg(results):
    n_total = sum(n for _, n in results)
    coef = sum(n * params[0] for params, n in results) / n_total
    intercept = sum(n * params[1] for params, n in results) / n_total
    return [coef, intercept]

def run_federated(X_train, y_train, X_test, y_test, client_indices, config, seed):
    global_params = [np.zeros((1, 30)), np.zeros(1)]
    history = []

    for rnd in range(config["rounds"]):
        results = []
        for k, idx in enumerate(client_indices):
            Xk, yk = X_train.iloc[idx], y_train.iloc[idx]
            results.append(
                local_train(global_params, Xk, yk, config, seed=seed + 1000 * rnd + k)
            )

        global_params = fedavg(results)

        model = make_model(seed)
        set_params(model, global_params)
        metrics = evaluate(model, X_test, y_test)
        history.append(metrics)

    return history