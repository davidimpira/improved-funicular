"""Causal (live-computable) features for the reversal-probability model.

Everything here is computable at the close of bar t using only bars <= t,
so the same code path serves both training and live scoring.

Features (see REPORT.md section 3 for why these):
  vol_surprise   log(bar volume / trailing mean volume of the SAME UTC hour,
                 prior 30 same-hour bars) -- volume anomaly vs the hour's norm
  rel_vol_3      log(mean of last 3 bars' volume / trailing 30-day mean volume)
                 -- recent volume build-up
  mom3           3-bar log return scaled by trailing hourly-return std
  dist_hi        log(close / max high of prior 24 bars)  (<=0 unless breakout)
  dist_lo        log(close / min low of prior 24 bars)   (>=0 unless breakdown)
  runlen         signed count of consecutive same-direction closes, capped +/-6
  h_1 .. h_23    UTC hour-of-day dummies (hour 0 is the reference)

Target (training only): a confirmed reversal extremum occurs at any of bars
t+1 .. t+3 (the extremum itself is only confirmable 3 bars later still, but
labels may look forward -- features may not).
"""

import numpy as np
import pandas as pd

VOL_WINDOW = 30 * 24   # trailing 30 days of hourly bars
HOUR_BASE_N = 30       # same-hour bars for the hour-specific volume baseline
DONCHIAN_N = 24
HORIZON = 3            # target: reversal within the next HORIZON bars

BASE_FEATURES = ["vol_surprise", "rel_vol_3", "mom3", "dist_hi", "dist_lo",
                 "runlen"]
HOUR_DUMMIES = [f"h_{h}" for h in range(1, 24)]
FEATURES = BASE_FEATURES + HOUR_DUMMIES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: hourly OHLCV indexed by UTC timestamp, oldest first."""
    df = df.copy()
    df["hour"] = df.index.hour
    logc = np.log(df["close"])
    df["logret"] = logc.diff()
    sigma = df["logret"].rolling(VOL_WINDOW, min_periods=24 * 7).std()

    vol30 = df["volume"].rolling(VOL_WINDOW, min_periods=24 * 7).mean()
    df["rel_vol_3"] = np.log(
        (df["volume"].rolling(3).mean() / vol30).clip(lower=1e-6)
    )
    hour_base = df.groupby("hour")["volume"].transform(
        lambda s: s.shift(1).rolling(HOUR_BASE_N, min_periods=10).mean()
    )
    df["vol_surprise"] = np.log((df["volume"] / hour_base).clip(lower=1e-6))

    df["mom3"] = (logc - logc.shift(HORIZON)) / (sigma * np.sqrt(HORIZON))

    hi = df["high"].rolling(DONCHIAN_N).max().shift(1)
    lo = df["low"].rolling(DONCHIAN_N).min().shift(1)
    df["dist_hi"] = np.log(df["close"] / hi)
    df["dist_lo"] = np.log(df["close"] / lo)

    sgn = np.sign(df["logret"]).fillna(0.0)
    grp = (sgn != sgn.shift()).cumsum()
    run = sgn.groupby(grp).cumcount() + 1
    df["runlen"] = (run * sgn).clip(-6, 6)

    for h in range(1, 24):
        df[f"h_{h}"] = (df["hour"] == h).astype(float)
    return df


def add_target(df: pd.DataFrame, reversal_flags: pd.Series) -> pd.DataFrame:
    """target[t] = 1 iff a confirmed reversal extremum occurs at t+1..t+HORIZON."""
    df = df.copy()
    rev = reversal_flags.astype(bool)
    fwd = pd.Series(False, index=df.index)
    for k in range(1, HORIZON + 1):
        fwd |= rev.shift(-k).fillna(False)
    df["target"] = fwd.astype(int)
    # bars whose full horizon isn't observable yet get no label
    df.loc[df.index[-HORIZON:], "target"] = np.nan
    return df


def predict_proba(row: pd.Series, model: dict) -> float:
    """Score one feature row with a saved logistic model {feature: coef}."""
    z = model["params"]["const"]
    for name in model["features"]:
        z += model["params"][name] * float(row[name])
    return float(1.0 / (1.0 + np.exp(-z)))
