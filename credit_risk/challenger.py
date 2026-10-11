"""Monotonic gradient-boosting challenger model (scikit-learn HistGradientBoosting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from .data import CATEGORICAL_FEATURES, MONOTONE_DIRECTION, NUMERIC_FEATURES


class GBMChallenger:
    def __init__(self, **params):
        self.params = dict(learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                           min_samples_leaf=200, l2_regularization=1.0,
                           early_stopping=True, validation_fraction=0.15, random_state=7)
        self.params.update(params)

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[NUMERIC_FEATURES].astype(float).copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = df[c].astype(str).map(self.cat_maps[c]).fillna(-1).astype(float)
        return X

    def fit(self, df: pd.DataFrame, y) -> "GBMChallenger":
        self.cat_maps = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))}
                         for c in CATEGORICAL_FEATURES}
        # Risk rises with +1 features; the model predicts P(default), so constraint = direction.
        cst = [MONOTONE_DIRECTION.get(f, 0) for f in NUMERIC_FEATURES] + [0] * len(CATEGORICAL_FEATURES)
        cat_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
        self.model = HistGradientBoostingClassifier(monotonic_cst=cst, categorical_features=cat_mask,
                                                    **self.params)
        self.model.fit(self._encode(df), np.asarray(y))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._encode(df))[:, 1]

    def importance(self, df: pd.DataFrame, y, n_repeats: int = 3) -> pd.Series:
        r = permutation_importance(self.model, self._encode(df), np.asarray(y), scoring="roc_auc",
                                   n_repeats=n_repeats, random_state=0)
        return pd.Series(r.importances_mean, index=NUMERIC_FEATURES + CATEGORICAL_FEATURES
                         ).sort_values(ascending=False)
