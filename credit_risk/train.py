"""End-to-end training: data -> out-of-time split -> scorecard + challenger -> report.

Usage:
    python -m credit_risk.train                      # synthetic data
    python -m credit_risk.train --csv loans.csv      # your own data (see README)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from . import metrics as M
from .challenger import GBMChallenger
from .data import DATE_COL, FEATURES, TARGET, generate_synthetic, load_csv, out_of_time_split
from .scorecard import Scorecard


def run(csv: str | None, out: Path, cutoff: str, seed: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    df = load_csv(csv) if csv else generate_synthetic(seed=seed)
    train, test = out_of_time_split(df, cutoff)
    ytr, yte = train[TARGET].to_numpy(), test[TARGET].to_numpy()

    sc = Scorecard().fit(train, ytr)
    gbm = GBMChallenger().fit(train, ytr)
    p_sc, p_gbm = sc.predict_proba(test), gbm.predict_proba(test)

    results = {
        "data": {"source": csv or "synthetic", "train_loans": len(train), "oot_loans": len(test),
                 "train_default_rate": float(ytr.mean()), "oot_default_rate": float(yte.mean()),
                 "oot_cutoff": cutoff},
        "scorecard_oot": M.summary(yte, p_sc),
        "gbm_challenger_oot": M.summary(yte, p_gbm),
        "scorecard_train": M.summary(ytr, sc.predict_proba(train)),
        "selected_features": sc.selected,
        "information_value": {f: round(b.iv, 4) for f, b in sorted(sc.bins.items(), key=lambda kv: -kv[1].iv)},
    }

    # Stability: score PSI and per-feature CSI, train vs. out-of-time.
    s_tr, s_te = sc.score(train), sc.score(test)
    results["score_psi"] = M.psi(s_tr, s_te)
    results["feature_csi"] = {f: round(M.psi(sc.bins[f].bin_index(train[f]), sc.bins[f].bin_index(test[f])), 4)
                              if sc.bins[f].kind == "numeric" else None for f in sc.selected}
    test = test.assign(score=s_te, month=test[DATE_COL].dt.to_period("M").astype(str))
    monthly = test.groupby("month").apply(lambda g: pd.Series({
        "loans": len(g), "default_rate": g[TARGET].mean(), "score_psi": M.psi(s_tr, g["score"])}),
        include_groups=False).reset_index()

    # Artifacts.
    joblib.dump({"scorecard": sc, "gbm": gbm}, out / "models.joblib")
    (out / "scorecard.json").write_text(sc.to_json())
    sc.table().to_csv(out / "scorecard_table.csv", index=False)
    M.decile_table(yte, p_sc).to_csv(out / "oot_decile_calibration.csv", index=False)
    monthly.to_csv(out / "oot_monthly_monitoring.csv", index=False)
    gbm.importance(test, yte).rename("auc_drop").to_csv(out / "gbm_permutation_importance.csv")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    _plots(out, yte, p_sc, p_gbm, s_tr, s_te, monthly)
    _report(out, results, sc, monthly)
    return results


def _plots(out, y, p_sc, p_gbm, s_tr, s_te, monthly):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for p, name in [(p_sc, "Scorecard"), (p_gbm, "GBM challenger")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {M.auc(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(title="ROC – out-of-time", xlabel="False positive rate", ylabel="True positive rate")
    ax[0].legend(loc="lower right")
    d = M.decile_table(y, p_sc)
    ax[1].plot(d["decile"].astype(int), d["predicted_pd"], "o-", label="Predicted PD")
    ax[1].plot(d["decile"].astype(int), d["actual_default_rate"], "s-", label="Actual default rate")
    ax[1].set(title="Calibration by risk decile (scorecard)", xlabel="Decile (10 = riskiest)", ylabel="Rate")
    ax[1].legend()
    ax[2].hist(s_tr, bins=40, alpha=0.6, density=True, label="Train")
    ax[2].hist(s_te, bins=40, alpha=0.6, density=True, label="Out-of-time")
    ax[2].set(title="Score distribution", xlabel="Score", ylabel="Density")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out / "model_performance.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(monthly["month"], monthly["score_psi"], color="#4C78A8")
    ax.axhline(0.10, ls="--", color="orange", label="Monitor (0.10)")
    ax.axhline(0.25, ls="--", color="red", label="Action (0.25)")
    ax.set(title="Monthly score PSI vs. development sample", ylabel="PSI")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "monitoring_psi.png", dpi=130)
    plt.close(fig)


def _report(out, r, sc, monthly):
    s, g = r["scorecard_oot"], r["gbm_challenger_oot"]
    lines = [
        "# Model Development Report – Consumer Loan PD",
        "",
        f"Data: **{r['data']['source']}** · train {r['data']['train_loans']:,} loans "
        f"(default rate {r['data']['train_default_rate']:.1%}) · out-of-time {r['data']['oot_loans']:,} loans "
        f"(default rate {r['data']['oot_default_rate']:.1%}), split at {r['data']['oot_cutoff']}.",
        "",
        "## Out-of-time performance",
        "",
        "| Model | AUC | Gini | KS | Brier |",
        "|---|---|---|---|---|",
        f"| WoE scorecard (champion) | {s['auc']:.3f} | {s['gini']:.3f} | {s['ks']:.3f} | {s['brier']:.4f} |",
        f"| Monotonic GBM (challenger) | {g['auc']:.3f} | {g['gini']:.3f} | {g['ks']:.3f} | {g['brier']:.4f} |",
        "",
        "![performance](model_performance.png)",
        "",
        "## Stability",
        "",
        f"Score PSI (train vs. out-of-time): **{r['score_psi']:.3f}** ({M.psi_label(r['score_psi'])}).",
        "",
        "![psi](monitoring_psi.png)",
        "",
        "## Selected features (Information Value)",
        "",
        "| Feature | IV |",
        "|---|---|",
    ] + [f"| {f} | {iv:.3f} |" for f, iv in r["information_value"].items() if f in sc.selected] + [
        "",
        "Full points table: `scorecard_table.csv`. Scale: 600 points at 50:1 good:bad odds, 20 points to double the odds.",
    ]
    (out / "MODEL_REPORT.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Path to a loan-level CSV (defaults to synthetic data)")
    ap.add_argument("--out", default="reports", help="Output folder")
    ap.add_argument("--cutoff", default="2024-07-01", help="Out-of-time split date")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    r = run(a.csv, Path(a.out), a.cutoff, a.seed)
    s, g = r["scorecard_oot"], r["gbm_challenger_oot"]
    print(f"Scorecard  OOT AUC {s['auc']:.3f}  Gini {s['gini']:.3f}  KS {s['ks']:.3f}")
    print(f"GBM        OOT AUC {g['auc']:.3f}  Gini {g['gini']:.3f}  KS {g['ks']:.3f}")
    print(f"Score PSI  {r['score_psi']:.3f} ({M.psi_label(r['score_psi'])})  -> reports in {a.out}/")


if __name__ == "__main__":
    main()
