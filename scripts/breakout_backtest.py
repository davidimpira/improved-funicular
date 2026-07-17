"""Backtest: hour-range breakout entries in the direction of a HTF bias.

The setup being tested (per the strategy owner):
  - A signal hour closes. If bias is LONG, place a buy stop just above that
    hour's high with a protective stop below its low. If SHORT, mirror it.
  - Bias proxy here: close vs the 200-hour SMA (swap in your own bias).
  - The entry order lives for TRIGGER_WINDOW bars, then is cancelled.
  - Exits: fixed R-multiple targets (1R/2R/3R, R = entry - protective stop)
    vs the protective stop, evaluated over the next MAX_HOLD bars; if
    neither is hit, exit at the last close (PnL in R).

Conservative path assumption: if a single hourly bar touches both target
and stop, the trade is counted as a LOSS (intra-bar order is unknowable
from OHLC).

Trades are event-study style (overlapping signals allowed) -- this measures
the per-hour edge of the setup, not portfolio equity.

Usage:
    python scripts/breakout_backtest.py --data data/btcusd_1h.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

SMA_N = 200          # bias proxy: close vs 200h SMA
TRIGGER_WINDOW = 3   # bars the entry stop-order stays live
MAX_HOLD = 48        # bars before time-exit at close
TARGETS = [1, 2, 3]  # R-multiple profit targets
FEE_SIDE = 0.0005    # 5 bps per side (fees + slippage), for the net column


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["sma"] = df["close"].rolling(SMA_N).mean()
    return df.dropna(subset=["sma"])


def simulate(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    long_bias = (df["close"] > df["sma"]).to_numpy()
    hours = df.index.hour.to_numpy()
    n = len(df)
    trades = []

    for t in range(n - TRIGGER_WINDOW - MAX_HOLD - 1):
        hi, lo = h[t], l[t]
        if hi <= lo:
            continue
        side = 1 if long_bias[t] else -1

        # entry trigger within the next TRIGGER_WINDOW bars
        te, entry = None, None
        for i in range(1, TRIGGER_WINDOW + 1):
            if side == 1 and h[t + i] > hi:
                te, entry = t + i, max(hi, o[t + i])
                break
            if side == -1 and l[t + i] < lo:
                te, entry = t + i, min(lo, o[t + i])
                break
        if te is None:
            continue

        stop = lo if side == 1 else hi
        risk = abs(entry - stop)
        if risk <= 0:
            continue

        rec = {"signal_ts": df.index[t], "signal_hour": int(hours[t]),
               "side": side, "entry_ts": df.index[te],
               "risk_pct": risk / entry}
        for k in TARGETS:
            target = entry + side * k * risk
            r_mult = None
            for j in range(te, min(te + MAX_HOLD, n)):
                lo_j = l[j] if j > te else min(l[j], entry)
                hi_j = h[j] if j > te else max(h[j], entry)
                hit_stop = lo_j <= stop if side == 1 else hi_j >= stop
                hit_tgt = hi_j >= target if side == 1 else lo_j <= target
                if hit_stop:          # conservative: stop wins ties
                    r_mult = -1.0
                    break
                if hit_tgt:
                    r_mult = float(k)
                    break
            if r_mult is None:        # time exit at close
                r_mult = side * (c[min(te + MAX_HOLD, n) - 1] - entry) / risk
            rec[f"r_{k}"] = r_mult
        trades.append(rec)
    return pd.DataFrame(trades)


def by_hour(trades: pd.DataFrame, fee_r: float) -> pd.DataFrame:
    g = trades.groupby("signal_hour")
    out = pd.DataFrame({
        "trades": g.size(),
        "trigger_rate": np.nan,  # filled by caller
        "win1R": g["r_1"].apply(lambda s: (s > 0).mean()),
        "win2R": g["r_2"].apply(lambda s: (s > 0).mean()),
        "exp1R": g["r_1"].mean(),
        "exp2R": g["r_2"].mean(),
        "exp3R": g["r_3"].mean(),
    })
    out["exp2R_net"] = out["exp2R"] - fee_r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/btcusd_1h.csv")
    ap.add_argument("--out", default="results/breakout_backtest_by_hour.csv")
    args = ap.parse_args()

    df = load(args.data)
    trades = simulate(df)

    # round-trip cost expressed in R units (fees / average risk per trade)
    fee_r = float(2 * FEE_SIDE / trades["risk_pct"].mean())

    tab = by_hour(trades, fee_r)
    signals_per_hour = df.index.hour.value_counts().sort_index()
    tab["trigger_rate"] = (tab["trades"] / signals_per_hour).round(3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tab.round(4).to_csv(args.out)

    overall = {k: trades[f"r_{k}"].mean() for k in TARGETS}
    print(f"{len(trades):,} triggered trades "
          f"({len(trades)/len(df):.1%} of bars) | "
          f"avg risk {trades['risk_pct'].mean():.2%} of price | "
          f"round-trip cost ~{fee_r:.2f}R")
    print("overall expectancy (R):",
          {f"{k}R": round(v, 3) for k, v in overall.items()})

    windows = {"US 12-17": range(12, 18), "Asia lull 02-08": range(2, 9),
               "00:00 anchor": [0], "other": None}
    used = set().union(*[set(v) for v in windows.values() if v])
    for name, hrs in windows.items():
        m = (~trades["signal_hour"].isin(list(used))
             if hrs is None else trades["signal_hour"].isin(list(hrs)))
        sub = trades[m]
        print(f"{name:>16}: n={len(sub):5d}  exp2R={sub['r_2'].mean():+.3f}  "
              f"exp2R_net={sub['r_2'].mean() - fee_r:+.3f}  "
              f"win2R={(sub['r_2'] > 0).mean():.1%}")
    print(f"per-hour table -> {args.out}")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
