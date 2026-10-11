import unittest

import numpy as np
import pandas as pd

from credit_risk import metrics as M
from credit_risk.binning import fit_numeric
from credit_risk.data import MONOTONE_DIRECTION, TARGET, generate_synthetic, out_of_time_split
from credit_risk.scorecard import Scorecard


class TestCreditRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_synthetic(n=20_000, seed=1)
        cls.train, cls.test = out_of_time_split(cls.df)
        cls.sc = Scorecard().fit(cls.train, cls.train[TARGET].to_numpy())

    def test_default_rate_is_realistic(self):
        self.assertTrue(0.08 < self.df[TARGET].mean() < 0.20)

    def test_out_of_time_split_has_no_overlap(self):
        self.assertLess(self.train["issue_date"].max(), self.test["issue_date"].min())

    def test_bins_are_monotone_in_expected_direction(self):
        for f, b in self.sc.bins.items():
            d = MONOTONE_DIRECTION.get(f, 0)
            if b.kind == "numeric" and d != 0 and len(b.bad_rate) > 1:
                self.assertTrue(np.all(np.diff(b.bad_rate) * d >= 0), f)

    def test_better_fico_never_scores_lower(self):
        base = self.test.iloc[[0]].copy()
        scores = [self.sc.score(base.assign(fico=v))[0] for v in (600, 650, 700, 750, 800)]
        self.assertEqual(scores, sorted(scores))

    def test_points_reproduce_model_probability(self):
        s = self.sc.score(self.test.head(500))
        p_model = self.sc.predict_proba(self.test.head(500))
        p_points = self.sc.score_to_pd(s)
        self.assertLess(np.max(np.abs(p_model - p_points)), 0.01)  # rounding of points only

    def test_reason_codes_sorted_and_limited(self):
        risky = {"fico": 610, "annual_income": 30000, "dti": 40, "revolving_util": 95,
                 "inquiries_6m": 5, "delinq_2yrs": 2, "emp_length_yrs": 0, "loan_amount": 35000,
                 "open_accounts": 5, "term": "60 months", "home_ownership": "RENT",
                 "purpose": "small_business"}
        rc = self.sc.reason_codes(risky)
        self.assertLessEqual(len(rc), 4)
        self.assertEqual(rc[0]["feature"], "fico")
        self.assertEqual([r["points_lost"] for r in rc], sorted([r["points_lost"] for r in rc], reverse=True))

    def test_model_beats_random_out_of_time(self):
        self.assertGreater(M.auc(self.test[TARGET], self.sc.predict_proba(self.test)), 0.70)

    def test_psi(self):
        x = np.random.default_rng(0).normal(size=5000)
        self.assertAlmostEqual(M.psi(x, x), 0.0, places=6)
        self.assertGreater(M.psi(x, x + 1.0), 0.25)

    def test_monotone_merge_handles_noise(self):
        rng = np.random.default_rng(3)
        x = pd.Series(rng.uniform(0, 100, 5000))
        y = rng.binomial(1, 0.1 + 0.002 * x)
        b = fit_numeric("x", x, y, direction=+1)
        self.assertTrue(np.all(np.diff(b.bad_rate) >= 0))


if __name__ == "__main__":
    unittest.main()
