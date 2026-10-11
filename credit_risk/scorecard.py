"""Points-based credit scorecard (WoE logistic regression) with adverse-action reason codes."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .binning import FeatureBins, fit_categorical, fit_numeric
from .data import CATEGORICAL_FEATURES, MONOTONE_DIRECTION, NUMERIC_FEATURES

# Plain-language adverse action reasons (ECOA / Reg B style).
REASON_TEXT = {
    "fico": "Credit score is lower than preferred",
    "annual_income": "Income is insufficient for amount of credit requested",
    "dti": "Debt-to-income ratio is too high",
    "revolving_util": "Proportion of revolving balances to credit limits is too high",
    "inquiries_6m": "Too many recent credit inquiries",
    "delinq_2yrs": "Recent delinquency on credit obligations",
    "emp_length_yrs": "Length of employment is insufficient",
    "loan_amount": "Requested loan amount is high relative to profile",
    "open_accounts": "Number of open accounts",
    "term": "Requested loan term carries higher risk",
    "home_ownership": "Housing status",
    "purpose": "Loan purpose carries higher risk",
}


class Scorecard:
    def __init__(self, pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 50.0,
                 min_iv: float = 0.02, C: float = 1.0):
        self.pdo, self.base_score, self.base_odds = pdo, base_score, base_odds
        self.min_iv, self.C = min_iv, C
        self.bins: dict[str, FeatureBins] = {}
        self.selected: list[str] = []
        self.points: dict[str, list[float]] = {}

    # ---- fitting -------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "Scorecard":
        y = np.asarray(y)
        for f in NUMERIC_FEATURES:
            self.bins[f] = fit_numeric(f, df[f], y, MONOTONE_DIRECTION.get(f, 0))
        for f in CATEGORICAL_FEATURES:
            self.bins[f] = fit_categorical(f, df[f], y)
        self.selected = [f for f, b in self.bins.items() if b.iv >= self.min_iv]

        X = self._woe_matrix(df)
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        self._build_points()
        return self

    def _woe_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.bins[f].transform(df[f]) for f in self.selected])

    def _build_points(self) -> None:
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        b0, coefs, n = self.model.intercept_[0], self.model.coef_[0], len(self.selected)
        self.factor, self.offset = factor, offset
        for f, b in zip(self.selected, coefs):
            woe = np.asarray(self.bins[f].woe)
            # log-odds(good) = -(b0 + sum b*woe); spread intercept/offset evenly over features.
            self.points[f] = list(np.round(-(b * woe + b0 / n) * factor + offset / n, 1))

    # ---- scoring -------------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._woe_matrix(df))[:, 1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return np.sum([np.asarray(self.points[f])[self.bins[f].bin_index(df[f])]
                       for f in self.selected], axis=0)

    def reason_codes(self, row: pd.Series | dict, top_n: int = 4) -> list[dict]:
        """Features costing the applicant the most points vs. the best attainable bin."""
        row = pd.Series(row)
        out = []
        for f in self.selected:
            pts = np.asarray(self.points[f])
            i = int(self.bins[f].bin_index(pd.Series([row[f]]))[0])
            shortfall = float(pts.max() - pts[i])
            if shortfall > 0:
                out.append({"feature": f, "reason": REASON_TEXT.get(f, f),
                            "points_lost": round(shortfall, 1)})
        return sorted(out, key=lambda r: -r["points_lost"])[:top_n]

    # ---- reporting / export --------------------------------------------------------------
    def table(self) -> pd.DataFrame:
        rows = []
        for f in self.selected:
            b = self.bins[f]
            for lab, cnt, br, w, p in zip(b.labels(), b.count, b.bad_rate, b.woe, self.points[f]):
                rows.append({"feature": f, "bin": lab, "count": cnt, "bad_rate": br,
                             "woe": w, "points": p, "iv": b.iv})
        return pd.DataFrame(rows)

    def to_json(self) -> str:
        """Export everything a lightweight scorer (e.g. the web demo) needs."""
        feats = []
        for f in self.selected:
            b = asdict(self.bins[f])
            feats.append({"name": f, "kind": b["kind"], "edges": b["edges"],
                          "categories": b["categories"], "labels": self.bins[f].labels(),
                          "points": self.points[f], "bad_rate": b["bad_rate"],
                          "iv": round(b["iv"], 4), "reason": REASON_TEXT.get(f, f)})
        return json.dumps({"pdo": self.pdo, "base_score": self.base_score,
                           "base_odds": self.base_odds, "factor": self.factor,
                           "offset": self.offset, "features": feats}, indent=2)

    def score_to_pd(self, score) -> np.ndarray:
        """Invert the score scale: score = offset + factor * ln(odds_good)."""
        odds_good = np.exp((np.asarray(score, float) - self.offset) / self.factor)
        return 1 / (1 + odds_good)
