"""Score one applicant with the trained scorecard and print adverse-action reasons.

Usage:
    python -m credit_risk.score --applicant '{"fico": 662, "annual_income": 48000, "dti": 31,
        "revolving_util": 78, "inquiries_6m": 3, "delinq_2yrs": 1, "emp_length_yrs": 2,
        "loan_amount": 18000, "open_accounts": 9, "term": "60 months",
        "home_ownership": "RENT", "purpose": "debt_consolidation"}'
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd


def score_applicant(applicant: dict, models_path: str = "reports/models.joblib") -> dict:
    sc = joblib.load(models_path)["scorecard"]
    row = pd.DataFrame([applicant])
    score = float(sc.score(row)[0])
    return {"score": round(score), "probability_of_default": float(sc.predict_proba(row)[0]),
            "reasons": sc.reason_codes(applicant)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--applicant", required=True, help="Applicant attributes as JSON")
    ap.add_argument("--models", default="reports/models.joblib")
    a = ap.parse_args()
    print(json.dumps(score_applicant(json.loads(a.applicant), a.models), indent=2))


if __name__ == "__main__":
    main()
