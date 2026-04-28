import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.utils import resample

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


# =========================
# 1) Métricas
# =========================
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["auc"] = roc_auc_score(y_true, y_prob)

    observed = np.sum(y_true)
    expected = np.sum(y_prob)
    metrics["OE"] = observed / expected if expected > 0 else np.nan

    metrics["brier"] = np.mean((y_true - y_prob) ** 2)

    return metrics


# =========================
# 2) Modelos de selección
# =========================

def fit_lasso(X, y, cv=5):
    model = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        Cs=10,
        cv=cv,
        scoring="roc_auc",
        max_iter=5000,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def fit_elasticnet(X, y, cv=5):
    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        Cs=10,
        cv=cv,
        scoring="roc_auc",
        max_iter=5000,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def get_selected_features(model, feature_names, tol=1e-6):
    coefs = model.coef_.ravel()
    return [feature_names[i] for i in range(len(coefs)) if abs(coefs[i]) > tol]


# =========================
# 3) Stability selection (MI)
# =========================

def stability_selection(imputed_datasets, target, method="lasso", threshold=0.7):
    feature_counts = {}

    for df in imputed_datasets:
        X = df.drop(columns=[target])
        y = df[target]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == "lasso":
            model = fit_lasso(X_scaled, y)
        elif method == "elasticnet":
            model = fit_elasticnet(X_scaled, y)
        else:
            raise ValueError("method must be 'lasso' or 'elasticnet'")

        selected = get_selected_features(model, X.columns)

        for f in selected:
            feature_counts[f] = feature_counts.get(f, 0) + 1

    m = len(imputed_datasets)
    final_features = [
        f for f, count in feature_counts.items()
        if count / m >= threshold
    ]

    return final_features, feature_counts


# =========================
# 4) Bootstrap evaluación
# =========================

def bootstrap_evaluation(imputed_datasets, target, features, n_boot=100):
    results = []

    for df in imputed_datasets:
        X = df[features]
        y = df[target]

        for b in range(n_boot):
            X_boot, y_boot = resample(X, y, replace=True)

            mask = ~X.index.isin(X_boot.index)
            X_test, y_test = X[mask], y[mask]

            if len(X_test) == 0:
                continue

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000))
            ])

            pipe.fit(X_boot, y_boot)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            m = compute_metrics(y_test, y_prob)
            results.append(m)

    return results


# =========================
# 5) Resumen (media + IC)
# =========================

def summarize(results):
    summary = {}
    for metric in results[0].keys():
        vals = [r[metric] for r in results]
        summary[metric] = {
            "mean": np.mean(vals),
            "ci_95": np.percentile(vals, [2.5, 97.5])
        }
    return summary


# =========================
# 6) MAIN COMPARISON
# =========================

def run_comparison(imputed_datasets, target):

    # --- 1. Todas las variables
    all_features = imputed_datasets[0].drop(columns=[target]).columns.tolist()

    # --- 2. LASSO
    lasso_features, lasso_counts = stability_selection(
        imputed_datasets, target, method="lasso"
    )

    # --- 3. Elastic Net
    enet_features, enet_counts = stability_selection(
        imputed_datasets, target, method="elasticnet"
    )

    print("N features:")
    print("All:", len(all_features))
    print("LASSO:", len(lasso_features))
    print("ElasticNet:", len(enet_features))

    # --- Evaluación
    res_all = bootstrap_evaluation(imputed_datasets, target, all_features)
    res_lasso = bootstrap_evaluation(imputed_datasets, target, lasso_features)
    res_enet = bootstrap_evaluation(imputed_datasets, target, enet_features)

    # --- Resumen
    summary = {
        "All": summarize(res_all),
        "LASSO": summarize(res_lasso),
        "ElasticNet": summarize(res_enet)
    }

    return summary, lasso_features, enet_features


imputed_datasets = [
    pd.read_csv(f"./database/imputation_mice/data_imp_{i}.csv")
    for i in range(1, 11)
]

summary, lasso_feats, enet_feats = run_comparison(
    imputed_datasets,
    target="recurrence"
)

import pprint
pprint.pprint(summary)