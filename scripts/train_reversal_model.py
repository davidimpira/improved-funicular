"""Train and walk-forward-validate the live reversal-probability model.

Walk-forward protocol: for each test year Y (2020..), fit a logistic
regression on all bars strictly before Y and score year Y out-of-sample.
Pooled out-of-sample predictions feed the calibration and threshold tables
in results/live_signal_backtest.md -- pick the alert threshold from that
table, not from intuition.

The shipped model (models/reversal_model.json) is refit on the full history.

Usage:
    python scripts/train_reversal_model.py --data data/btcusd_1h.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from analysis import flag_reversals, load
from live_features import FEATURES, HORIZON, add_target, build_features


def fit_logit(train: pd.DataFrame):
    X = sm.add_constant(train[FEATURES].astype(float))
    return sm.Logit(train["target"].astype(int), X).fit(disp=0, maxiter=200)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df.index.year.unique())
    oos = []
    for year in years:
        train = df[df.index.year < year]
        test = df[df.index.year == year]
        if len(train) < 24 * 365 or train["target"].sum() < 200 or not len(test):
            continue
        model = fit_logit(train)
        Xt = sm.add_constant(test[FEATURES].astype(float), has_constant="add")
        oos.append(pd.DataFrame({
            "p": model.predict(Xt), "y": test["target"].astype(int),
            "year": year}, index=test.index))
    return pd.concat(oos)


def calibration_table(oos: pd.DataFrame) -> pd.DataFrame:
    dec = pd.qcut(oos["p"], 10, labels=False, duplicates="drop")
    tab = oos.groupby(dec).agg(mean_pred=("p", "mean"),
                               actual_rate=("y", "mean"),
                               n=("y", "size"))
    tab.index.name = "decile"
    return tab


def threshold_table(oos: pd.DataFrame, bars_per_week: float) -> pd.DataFrame:
    base = oos["y"].mean()
    rows = []
    for mult in [1.5, 2.0, 2.5, 3.0, 4.0]:
        thr = base * mult
        hit = oos["p"] >= thr
        n_alert = int(hit.sum())
        precision = oos.loc[hit, "y"].mean() if n_alert else np.nan
        rows.append({
            "lift_threshold": mult,
            "prob_threshold": round(thr, 4),
            "alerts_per_week": round(n_alert / (len(oos) / bars_per_week), 2),
            "precision": round(float(precision), 4) if n_alert else None,
            "realized_lift": round(float(precision / base), 2) if n_alert else None,
            "recall": round(float(oos.loc[hit, "y"].sum() / oos["y"].sum()), 4),
        })
    return pd.DataFrame(rows)


def auc(oos: pd.DataFrame) -> float:
    pos = oos.loc[oos["y"] == 1, "p"]
    neg = oos.loc[oos["y"] == 0, "p"]
    u = stats.mannwhitneyu(pos, neg, alternative="greater").statistic
    return float(u / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/btcusd_1h.csv")
    ap.add_argument("--model-out", default="models/reversal_model.json")
    ap.add_argument("--report-out", default="results/live_signal_backtest.md")
    args = ap.parse_args()

    raw = load(args.data)
    feats = build_features(raw[["open", "high", "low", "close", "volume"]])
    feats = add_target(feats, flag_reversals(raw))
    df = feats.dropna(subset=FEATURES + ["target"])

    oos = walk_forward(df)
    base = float(oos["y"].mean())
    cal = calibration_table(oos)
    thr = threshold_table(oos, bars_per_week=24 * 7)
    model_auc = auc(oos)
    brier = float(((oos["p"] - oos["y"]) ** 2).mean())
    brier_null = float(((base - oos["y"]) ** 2).mean())

    final = fit_logit(df)
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "w") as f:
        json.dump({
            "kind": "logistic",
            "target": f"confirmed reversal within next {HORIZON} bars",
            "features": FEATURES,
            "params": {k: float(v) for k, v in final.params.items()},
            "base_rate": base,
            "trained_through": str(df.index.max()),
            "n_train": int(len(df)),
            "oos": {"auc": model_auc, "brier": brier,
                    "brier_null": brier_null,
                    "years": sorted(oos["year"].unique().tolist())},
        }, f, indent=2)

    os.makedirs(os.path.dirname(args.report_out), exist_ok=True)
    per_year = oos.groupby("year").apply(
        lambda g: pd.Series({"auc": auc(g), "base_rate": g["y"].mean()}),
        include_groups=False)
    with open(args.report_out, "w") as f:
        f.write(
            "# Walk-forward backtest of the live reversal signal\n\n"
            f"Out-of-sample: {oos.index.min():%Y-%m-%d} to "
            f"{oos.index.max():%Y-%m-%d} ({len(oos):,} bars), each year scored "
            "by a model trained only on prior years.\n\n"
            f"Target: confirmed reversal within the next {HORIZON} bars. "
            f"Base rate: **{base:.2%}**.\n\n"
            f"- AUC: **{model_auc:.3f}**\n"
            f"- Brier score: {brier:.4f} (always-predict-base-rate: "
            f"{brier_null:.4f})\n\n"
            "## Calibration (out-of-sample deciles)\n\n"
            + cal.round(4).to_markdown() +
            "\n\n## Alert-threshold trade-off (pick your knob here)\n\n"
            + thr.to_markdown(index=False) +
            "\n\n`lift_threshold` = alert when predicted probability >= that "
            "multiple of the base rate. `precision` = fraction of alerts "
            "followed by a confirmed reversal within 3 bars.\n\n"
            "## Per-year stability\n\n"
            + per_year.round(4).to_markdown() + "\n")

    print(f"OOS AUC {model_auc:.3f} | base {base:.2%} | "
          f"brier {brier:.4f} vs null {brier_null:.4f}")
    print(thr.to_string(index=False))
    print(f"model -> {args.model_out}\nreport -> {args.report_out}")


if __name__ == "__main__":
    main()
