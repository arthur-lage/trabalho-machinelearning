import pandas as pd
import numpy as np
from config import CONFIG
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"

def load_and_split(seed):
    df = pd.read_csv(DATA_PATH, index_col=False)
    X = df.drop(columns="Class")
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y, test_size=CONFIG["test_size"])

    X_train, X_test = X_train.copy(), X_test.copy()
    time_scaler = StandardScaler().fit(X_train[["Time"]])
    amount_scaler = RobustScaler().fit(X_train[["Amount"]])

    for X_ in (X_train, X_test):
        X_["Time"] = time_scaler.transform(X_[["Time"]])
        X_["Amount"] = amount_scaler.transform(X_[["Amount"]])

    return X_train, X_test, y_train, y_test

def dirichlet_partition(y_train, n_clients, alpha, seed, min_size=10, max_tries=1000):
    rng = np.random.default_rng(seed)
    y = np.asarray(y_train)

    for _ in range(max_tries):
        client_indices = [[] for _ in range(n_clients)]

        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)

            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, part in enumerate(np.split(idx, cuts)):
                client_indices[k].extend(part.tolist())

        if min(len(ci) for ci in client_indices) >= min_size:
            return [np.array(sorted(ci), dtype=int) for ci in client_indices]

    raise RuntimeError(f"Sem partição válida após {max_tries} tentativas (alpha={alpha})")

def summarize_partition(y_train, client_indices):
    y = np.asarray(y_train)
    for k, idx in enumerate(client_indices):
        n_fraud = int(y[idx].sum())
        print(f"cliente {k}: total={len(idx):6d}  fraudes={n_fraud:4d}  legítimas={len(idx) - n_fraud:6d}")
