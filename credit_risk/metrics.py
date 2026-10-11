"""Model performance and stability metrics used in credit risk."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` vs `expected` using expected's deciles."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=len(edges) + 1)
    a = np.bincount(np.searchsorted(edges, actual, side="right"), minlength=len(edges) + 1)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    return "stable" if value < 0.10 else ("monitor" if value < 0.25 else "significant shift")


def decile_table(y, p) -> pd.DataFrame:
    """Predicted vs. actual default rate by risk decile (10 = riskiest)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), 10, labels=range(1, 11))
    t = df.groupby("decile", observed=True).agg(loans=("y", "size"), predicted_pd=("p", "mean"),
                                                actual_default_rate=("y", "mean"))
    return t.reset_index()


def summary(y, p) -> dict:
    return {"auc": auc(y, p), "gini": gini(y, p), "ks": ks(y, p),
            "brier": float(brier_score_loss(y, p)), "default_rate": float(np.mean(y)),
            "mean_pd": float(np.mean(p))}
