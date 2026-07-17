"""Hour-of-day inflection analysis for BTC/USD hourly candles.

Answers two questions:
  1. Are price inflections (reversals / breakouts) more likely at particular
     UTC clock hours?
  2. Is any hourly pattern purely clock-driven, or is it mediated by volume?

Event definitions (hourly bars, all times UTC, hour = bar open hour):

  REVERSAL  bar t is a confirmed local extremum of closes over [t-K, t+K]
            (K=3) AND both the move into and out of the extremum exceed
            0.75 x the trailing 30-day std of hourly log returns (noise
            filter, adapts to volatility regime).

  BREAKOUT  close[t] exceeds the max high (or min low) of the prior 24 bars
            (Donchian-24 breakout, i.e. a new daily-range extreme).

Statistics:
  - chi-square goodness-of-fit of event counts across the 24 hours
  - per-hour two-sided binomial tests vs the pooled rate, BH/FDR corrected
  - day-level block bootstrap (resample calendar days) for 95% CIs
  - split-sample stability (first half vs second half, Spearman rank corr)
  - clock vs volume: logistic regressions (hour dummies / relative volume /
    both) compared by likelihood-ratio tests and pseudo-R2, plus event rates
    within relative-volume terciles per hour.

Usage:
    python scripts/analysis.py --data data/btcusd_1h.csv --outdir results
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

K = 3                 # confirmation window (bars) each side of an extremum
NOISE_MULT = 0.75     # min move size, in units of trailing hourly-return std
VOL_WINDOW = 30 * 24  # trailing window for return std and volume baseline
DONCHIAN_N = 24       # lookback for breakout channel
BOOT_REPS = 2000
SEED = 20260717


# ---------------------------------------------------------------- data prep

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["hour"] = df.index.hour
    df["date"] = df.index.date
    df["logret"] = np.log(df["close"]).diff()
    df["ret_std"] = df["logret"].rolling(VOL_WINDOW, min_periods=24 * 7).std()
    # relative volume: this bar vs trailing 30-day mean (removes regime trend)
    df["rel_vol"] = df["volume"] / df["volume"].rolling(
        VOL_WINDOW, min_periods=24 * 7
    ).mean()
    df["log_rel_vol"] = np.log(df["rel_vol"].clip(lower=1e-6))
    # volume known BEFORE the bar: mean relative volume of the prior 3 bars
    df["log_rel_vol_lag"] = np.log(
        df["rel_vol"].shift(1).rolling(3).mean().clip(lower=1e-6)
    )
    # relative range: intrabar range vs trailing mean range (volatility control)
    rng = (df["high"] - df["low"]) / df["close"]
    df["rel_range"] = rng / rng.rolling(VOL_WINDOW, min_periods=24 * 7).mean()
    return df


# ------------------------------------------------------------ event flags

def flag_reversals(df: pd.DataFrame) -> pd.Series:
    c = df["close"].to_numpy()
    n = len(c)
    is_ext = np.zeros(n, dtype=bool)
    logc = np.log(c)
    thresh = (NOISE_MULT * df["ret_std"] * np.sqrt(K)).to_numpy()
    for t in range(K, n - K):
        win = c[t - K : t + K + 1]
        if not (c[t] == win.max() or c[t] == win.min()):
            continue
        move_in = abs(logc[t] - logc[t - K])
        move_out = abs(logc[t + K] - logc[t])
        th = thresh[t]
        if np.isnan(th):
            continue
        if move_in >= th and move_out >= th:
            is_ext[t] = True
    return pd.Series(is_ext, index=df.index, name="reversal")


def flag_breakouts(df: pd.DataFrame) -> pd.Series:
    hi = df["high"].rolling(DONCHIAN_N).max().shift(1)
    lo = df["low"].rolling(DONCHIAN_N).min().shift(1)
    return ((df["close"] > hi) | (df["close"] < lo)).rename("breakout")


# ------------------------------------------------------------- statistics

def hourly_table(df: pd.DataFrame, event: str) -> pd.DataFrame:
    g = df.groupby("hour")[event].agg(["sum", "count"])
    g.columns = ["events", "bars"]
    g["rate"] = g["events"] / g["bars"]
    return g


def chi_square(tab: pd.DataFrame) -> tuple[float, float]:
    expected = tab["bars"] / tab["bars"].sum() * tab["events"].sum()
    chi2, p = stats.chisquare(tab["events"], expected)
    return float(chi2), float(p)


def per_hour_binomial(tab: pd.DataFrame) -> pd.DataFrame:
    pooled = tab["events"].sum() / tab["bars"].sum()
    pvals = [
        stats.binomtest(int(r.events), int(r.bars), pooled).pvalue
        for r in tab.itertuples()
    ]
    tab = tab.copy()
    tab["pooled_rate"] = pooled
    tab["lift"] = tab["rate"] / pooled
    tab["p_raw"] = pvals
    tab["p_fdr"] = stats.false_discovery_control(pvals)
    return tab


def day_block_bootstrap(df: pd.DataFrame, event: str) -> pd.DataFrame:
    """Resample whole calendar days to get CIs that respect intraday dependence."""
    rng = np.random.default_rng(SEED)
    pivot_e = df.pivot_table(index="date", columns="hour", values=event,
                             aggfunc="sum").fillna(0).to_numpy()
    pivot_n = df.pivot_table(index="date", columns="hour", values=event,
                             aggfunc="count").fillna(0).to_numpy()
    ndays = pivot_e.shape[0]
    rates = np.empty((BOOT_REPS, 24))
    for b in range(BOOT_REPS):
        idx = rng.integers(0, ndays, ndays)
        e = pivot_e[idx].sum(axis=0)
        n = pivot_n[idx].sum(axis=0)
        rates[b] = np.divide(e, n, out=np.zeros(24), where=n > 0)
    lo, hi = np.percentile(rates, [2.5, 97.5], axis=0)
    return pd.DataFrame({"ci_lo": lo, "ci_hi": hi}, index=range(24))


def split_stability(df: pd.DataFrame, event: str) -> dict:
    mid = df.index[len(df) // 2]
    a = hourly_table(df[df.index < mid], event)["rate"]
    b = hourly_table(df[df.index >= mid], event)["rate"]
    rho, p = stats.spearmanr(a, b)
    return {"spearman_rho": float(rho), "p": float(p),
            "split_at": str(mid), "first_half": a.to_dict(),
            "second_half": b.to_dict()}


def clock_vs_volume(df: pd.DataFrame, event: str,
                    vol_col: str = "log_rel_vol") -> dict:
    """Nested logistic regressions: hour dummies vs volume vs both.

    vol_col="log_rel_vol" uses the event bar's own volume (contemporaneous —
    partly mechanical, since inflection bars attract volume). Pass a lagged
    column to test whether volume known BEFORE the bar carries the signal.
    """
    import statsmodels.api as sm

    d = df.dropna(subset=[vol_col, event]).copy()
    y = d[event].astype(int)
    hour_d = pd.get_dummies(d["hour"], prefix="h", drop_first=True).astype(float)
    volu = sm.add_constant(d[[vol_col]])
    hour_X = sm.add_constant(hour_d)
    both_X = sm.add_constant(pd.concat([hour_d, d[[vol_col]]], axis=1))

    def fit(X):
        return sm.Logit(y, X).fit(disp=0)

    m_hour, m_vol, m_both = fit(hour_X), fit(volu), fit(both_X)
    null_ll = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0).llf

    def lr(full, restricted, df_diff):
        stat = 2 * (full.llf - restricted.llf)
        return {"lr_stat": float(stat),
                "p": float(stats.chi2.sf(stat, df_diff))}

    def pr2(m):
        return float(1 - m.llf / null_ll)

    return {
        "n": int(len(y)),
        "volume_measure": vol_col,
        "pseudo_r2": {"hour_only": pr2(m_hour), "volume_only": pr2(m_vol),
                      "hour_plus_volume": pr2(m_both)},
        "hour_effect_after_volume": lr(m_both, m_vol, 23),
        "volume_effect_after_hour": lr(m_both, m_hour, 1),
        "volume_coef_alone": float(m_vol.params[vol_col]),
        "volume_coef_with_hours": float(m_both.params[vol_col]),
    }


def volume_strata(df: pd.DataFrame, event: str) -> pd.DataFrame:
    d = df.dropna(subset=["rel_vol", event]).copy()
    d["vol_tercile"] = pd.qcut(d["rel_vol"], 3, labels=["low", "mid", "high"])
    out = d.pivot_table(index="hour", columns="vol_tercile", values=event,
                        aggfunc="mean", observed=True)
    return out


# ---------------------------------------------------------------- plotting

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#008300",
           "violet": "#4a3aa7", "ink": "#0b0b0b", "muted": "#52514e",
           "grid": "#e5e4e0"}


def style_ax(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=PALETTE["ink"])
    ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["muted"])
    ax.set_xlabel("UTC hour of bar open", fontsize=9, color=PALETTE["muted"])
    ax.set_xticks(range(0, 24, 2))
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)


def plot_rate(tab, boot, pooled, title, fname, color):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    hours = tab.index.to_numpy()
    ax.bar(hours, tab["rate"], color=color, width=0.7)
    err_lo = tab["rate"] - boot["ci_lo"].to_numpy()
    err_hi = boot["ci_hi"].to_numpy() - tab["rate"]
    ax.errorbar(hours, tab["rate"], yerr=[err_lo, err_hi], fmt="none",
                ecolor=PALETTE["ink"], elinewidth=1, capsize=2, alpha=0.6)
    ax.axhline(pooled, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.annotate(f"pooled {pooled:.2%}", xy=(23.4, pooled),
                fontsize=8, color=PALETTE["muted"], va="bottom", ha="right")
    style_ax(ax, title, "event rate per bar")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_volume_curve(df, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby("hour")["rel_vol"].mean()
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.plot(g.index, g.values, color=PALETTE["blue"], linewidth=2,
            marker="o", markersize=4)
    ax.axhline(1.0, color=PALETTE["muted"], linestyle="--", linewidth=1)
    style_ax(ax, "Mean relative volume by UTC hour (1.0 = 30-day average)",
             "relative volume")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def plot_strata(strata, pooled_by_terc, title, fname):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    colors = {"low": "#9db9dd", "mid": PALETTE["blue"], "high": "#123c6e"}
    for col in ["low", "mid", "high"]:
        ax.plot(strata.index, strata[col], color=colors[col], linewidth=2,
                label=f"{col} volume (avg {pooled_by_terc[col]:.2%})")
    ax.legend(fontsize=8, frameon=False)
    style_ax(ax, title, "event rate per bar")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/btcusd_1h.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load(args.data)
    df["reversal"] = flag_reversals(df)
    df["breakout"] = flag_breakouts(df)

    summary: dict = {
        "data": {
            "source": "Bitstamp BTC/USD 1h OHLCV",
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "bars": int(len(df)),
            "params": {"K": K, "noise_mult": NOISE_MULT,
                       "donchian_n": DONCHIAN_N,
                       "bootstrap_reps": BOOT_REPS},
        }
    }

    for event, color in [("reversal", PALETTE["blue"]),
                         ("breakout", PALETTE["orange"])]:
        tab = per_hour_binomial(hourly_table(df, event))
        chi2, chi_p = chi_square(tab)
        boot = day_block_bootstrap(df, event)
        stab = split_stability(df, event)
        cvv = clock_vs_volume(df, event)
        cvv_lag = clock_vs_volume(df, event, vol_col="log_rel_vol_lag")
        strata = volume_strata(df, event)

        pooled = tab["pooled_rate"].iloc[0]
        plot_rate(tab, boot, pooled,
                  f"BTC/USD hourly {event} rate by UTC hour "
                  f"(95% day-bootstrap CI)",
                  os.path.join(args.outdir, f"{event}_rate_by_hour.png"),
                  color)
        d_nonan = df.dropna(subset=["rel_vol"])
        terc = pd.qcut(d_nonan["rel_vol"], 3, labels=["low", "mid", "high"])
        pooled_terc = d_nonan.groupby(terc, observed=True)[event].mean().to_dict()
        plot_strata(strata, pooled_terc,
                    f"{event.capitalize()} rate by UTC hour within "
                    f"relative-volume terciles",
                    os.path.join(args.outdir, f"{event}_by_hour_volume_terciles.png"))

        tab_out = tab.copy()
        tab_out[["ci_lo", "ci_hi"]] = boot[["ci_lo", "ci_hi"]].to_numpy()
        tab_out.to_csv(os.path.join(args.outdir, f"{event}_hourly_table.csv"))

        summary[event] = {
            "total_events": int(tab["events"].sum()),
            "pooled_rate": float(pooled),
            "chi2_uniform": {"stat": chi2, "p": chi_p},
            "top_hours": tab.sort_values("lift", ascending=False)
                            .head(5)[["events", "rate", "lift", "p_fdr"]]
                            .round(4).to_dict("index"),
            "bottom_hours": tab.sort_values("lift")
                               .head(5)[["events", "rate", "lift", "p_fdr"]]
                               .round(4).to_dict("index"),
            "sig_hours_fdr05": tab.index[tab["p_fdr"] < 0.05].tolist(),
            "split_stability": {k: stab[k] for k in
                                ["spearman_rho", "p", "split_at"]},
            "clock_vs_volume_contemporaneous": cvv,
            "clock_vs_volume_lagged": cvv_lag,
        }

    plot_volume_curve(df, os.path.join(args.outdir, "volume_by_hour.png"))
    vol_by_hour = df.groupby("hour")["rel_vol"].mean()
    summary["volume_curve"] = vol_by_hour.round(3).to_dict()

    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
