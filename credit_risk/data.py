"""Data loading and a synthetic consumer-loan generator.

The synthetic data mimics the structure of public consumer-lending datasets
(e.g. LendingClub): borrower credit attributes, loan terms and an issue date
for out-of-time validation. Default risk is driven by known relationships
(higher DTI / utilization / inquiries and lower FICO / income raise risk), so
model behaviour can be checked against ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "default"
DATE_COL = "issue_date"

NUMERIC_FEATURES = [
    "fico", "annual_income", "dti", "revolving_util", "inquiries_6m",
    "delinq_2yrs", "emp_length_yrs", "loan_amount", "open_accounts",
]
CATEGORICAL_FEATURES = ["term", "home_ownership", "purpose"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Expected direction of each numeric feature vs. default risk (+1 = higher value, more risk).
MONOTONE_DIRECTION = {
    "fico": -1, "annual_income": -1, "dti": +1, "revolving_util": +1,
    "inquiries_6m": +1, "delinq_2yrs": +1, "emp_length_yrs": -1,
    "loan_amount": +1, "open_accounts": 0,
}


def generate_synthetic(n: int = 60_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic loan book with a realistic ~12-14% default rate."""
    rng = np.random.default_rng(seed)
    issue = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    t = (issue - issue.min()).days.values / 730.0  # 0 -> 1 across the two-year window
    # Applicant mix drifts riskier over time (lower FICO, higher utilization) for monitoring demos.
    fico = np.clip(rng.normal(702 - 16 * t, 45, n), 580, 850).round()
    income = np.clip(rng.lognormal(np.log(72_000), 0.5, n), 15_000, 600_000).round(-2)
    dti = np.clip(rng.gamma(4.0, 4.5, n) + (720 - fico) * 0.03, 0, 60).round(1)
    util = np.clip(rng.beta(2, 2.6, n) * 100 + (700 - fico) * 0.15 + 6 * t, 0, 120).round(1)
    inquiries = rng.poisson(np.clip(0.8 + (700 - fico) / 120, 0.1, None))
    delinq = rng.poisson(np.clip(0.25 + (680 - fico) / 300, 0.02, None))
    emp = np.clip(rng.gamma(2.0, 3.0, n), 0, 30).round()
    loan_amt = np.clip(rng.lognormal(np.log(14_000), 0.6, n), 1_000, 40_000).round(-2)
    open_acc = rng.poisson(11, n)
    term = rng.choice(["36 months", "60 months"], n, p=[0.72, 0.28])
    home = rng.choice(["MORTGAGE", "RENT", "OWN"], n, p=[0.48, 0.40, 0.12])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"],
        n, p=[0.55, 0.22, 0.08, 0.04, 0.11])

    # Latent log-odds of default (ground truth).
    z = (-3.0
         - 0.020 * (fico - 695)
         - 0.45 * (np.log(income) - np.log(72_000))
         + 0.035 * (dti - 18)
         + 0.012 * (util - 45)
         + 0.16 * inquiries
         + 0.30 * np.minimum(delinq, 3)
         - 0.03 * np.minimum(emp, 10)
         + 0.25 * (np.log(loan_amt) - np.log(14_000))
         + 0.45 * (term == "60 months")
         + 0.15 * (home == "RENT") - 0.10 * (home == "OWN")
         + 0.55 * (purpose == "small_business") + 0.10 * (purpose == "other")
         - 0.10 * (purpose == "credit_card")
         + 0.0009 * (dti - 18) ** 2  # mild non-linearity
         + rng.normal(0, 0.9, n)     # risk not captured by bureau data (keeps AUC realistic)
         )
    pd_true = 1 / (1 + np.exp(-z))
    default = rng.binomial(1, pd_true)

    return pd.DataFrame({
        "fico": fico, "annual_income": income, "dti": dti, "revolving_util": util,
        "inquiries_6m": inquiries, "delinq_2yrs": delinq, "emp_length_yrs": emp,
        "loan_amount": loan_amt, "open_accounts": open_acc, "term": term,
        "home_ownership": home, "purpose": purpose, DATE_COL: issue, TARGET: default,
    })


def load_csv(path: str) -> pd.DataFrame:
    """Load a real loan file with the same column names (see README for mapping)."""
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    missing = [c for c in FEATURES + [TARGET, DATE_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def out_of_time_split(df: pd.DataFrame, cutoff: str = "2024-07-01"):
    """Train on loans issued before `cutoff`, validate on loans issued after."""
    cut = pd.Timestamp(cutoff)
    return df[df[DATE_COL] < cut].copy(), df[df[DATE_COL] >= cut].copy()
