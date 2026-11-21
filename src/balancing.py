import numpy as np

def adaptive_undersample(X, y, tau, r, n_piso, seed):
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y)

    idx_fraud = np.where(y_arr == 1)[0]
    idx_legit = np.where(y_arr == 0)[0]
    n_min = len(idx_fraud)

    # 1) poucas fraudes: suspende, devolve tudo intacto
    if n_min < tau:
        return X, y

    # 2) quantas legítimas manter
    n_keep = max(r * n_min, n_piso)

    # 3) já tem menos legítimas que o alvo: não descarta nada
    if len(idx_legit) <= n_keep:
        return X, y

    # 4) sorteia as legítimas que ficam; todas as fraudes ficam
    kept_legit = rng.choice(idx_legit, size=n_keep, replace=False)
    keep = np.sort(np.concatenate([idx_fraud, kept_legit]))

    return X.iloc[keep], y.iloc[keep]