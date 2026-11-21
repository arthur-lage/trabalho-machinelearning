import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.linear_model import SGDClassifier

def make_model(seed):
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-4,
        learning_rate="constant", eta0=0.01, random_state=seed,
    )

def get_params(model):
    return [model.coef_, model.intercept_]

def set_params(model, params):
    model.coef_, model.intercept_ = params[0].copy(), params[1].copy()
    model.classes_ = np.array([0, 1])

def evaluate(model, X_test, y_test):
    X_test = X_test.to_numpy()
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auprc": average_precision_score(y_test, y_score),
    }