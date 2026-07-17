"""Live scorer: alert on unusual volume for the hour (+ reversal context).

Designed to run from cron (e.g. GitHub Actions, a few minutes past each
hour). Fetches the last ~40 days of hourly BTC/USD candles from Bitstamp,
rebuilds the causal features, and evaluates the last CLOSED bar.

PRIMARY trigger — volume surprise: the bar's volume vs the trailing mean
volume of the same UTC hour (prior 30 same-hour bars). The breakout
backtest (results/breakout_backtest_by_hour.csv, scripts/breakout_backtest.py)
showed this is the condition that separates hour-range breakouts with
positive expectancy from noise; hour-of-day alone does not.

SECONDARY context — the reversal model's P(confirmed reversal within 3
bars), included in every alert and able to trigger one on its own if
ALERT_MIN_LIFT is set.

Alert rule (stateless, no DB needed): fire only on upward threshold
crossings (previous bar below, current bar at/above), so a stretch of
consecutive hot hours produces one alert, not one per hour.

The alert message includes the signal bar's high/low — the entry/stop
levels for a bias-direction range-breakout tactic.

Environment:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   if unset, prints instead of sending
  VOL_SURPRISE_MIN   volume multiple of the hour's norm that triggers an
                     alert (default 2.0; backtest edge grows toward 3.0
                     with fewer signals)
  ALERT_MIN_LIFT     optional reversal-model trigger, as a multiple of the
                     base rate (default 0 = model is context only; 2.5 was
                     the earlier default, see results/live_signal_backtest.md)

Usage:
    python scripts/live_score.py [--dry-run]
"""

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

from live_features import FEATURES, build_features, predict_proba

API = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
LOOKBACK_BARS = 960  # 40 days; single API call, enough for all rolling windows


def fetch_recent() -> pd.DataFrame:
    params = urllib.parse.urlencode({"step": 3600, "limit": LOOKBACK_BARS})
    req = urllib.request.Request(
        f"{API}?{params}", headers={"User-Agent": "btc-reversal-alert/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                rows = json.load(resp)["data"]["ohlc"]
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df = df.sort_values("ts").drop_duplicates("ts").set_index("ts")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    # drop the still-forming bar: keep only bars whose hour has fully elapsed
    now = dt.datetime.now(dt.timezone.utc)
    return df[df.index + dt.timedelta(hours=1) <= now]


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set; message below:")
        print(text)
        return False
    data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    with urllib.request.urlopen(req, timeout=30) as resp:
        ok = json.load(resp).get("ok", False)
    print(f"telegram send ok={ok}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="never send, just print")
    ap.add_argument("--model", default="models/reversal_model.json")
    args = ap.parse_args()

    with open(args.model) as f:
        model = json.load(f)
    base = model["base_rate"]
    vol_min = float(os.environ.get("VOL_SURPRISE_MIN", "2.0"))
    min_lift = float(os.environ.get("ALERT_MIN_LIFT", "0"))

    feats = build_features(fetch_recent())
    scored = feats.dropna(subset=FEATURES)
    if len(scored) < 2:
        print("not enough complete bars with features; aborting")
        sys.exit(1)

    last, prev = scored.iloc[-1], scored.iloc[-2]
    vol_last = math.exp(last["vol_surprise"])
    vol_prev = math.exp(prev["vol_surprise"])
    p_last = predict_proba(last, model)
    p_prev = predict_proba(prev, model)

    bar_ts = scored.index[-1]
    print(f"bar {bar_ts:%Y-%m-%d %H:%M} UTC | close {last['close']:,.0f} | "
          f"volume {vol_last:.2f}x hour norm (prev {vol_prev:.2f}x, "
          f"threshold {vol_min:.1f}x) | P(reversal<=3 bars) {p_last:.1%}")

    vol_crossing = vol_last >= vol_min and vol_prev < vol_min
    rev_crossing = (min_lift > 0 and p_last >= base * min_lift
                    and p_prev < base * min_lift)
    if not vol_crossing and not rev_crossing:
        print("no alert (below threshold or already alerted on this stretch)")
        return

    trigger = ("UNUSUAL VOLUME for the hour" if vol_crossing
               else "reversal-model threshold")
    msg = (
        f"BTC/USD {trigger} — {bar_ts:%H:%M} UTC bar\n"
        f"Volume: {vol_last:.1f}x this hour's 30-day norm\n"
        f"Close: {last['close']:,.0f}\n"
        f"Signal-bar range (breakout entry/stop levels):\n"
        f"  high {last['high']:,.0f} / low {last['low']:,.0f} "
        f"(width {(last['high'] - last['low']) / last['close']:.2%})\n"
        f"Context: P(reversal<=3 bars) {p_last:.1%} "
        f"({p_last / base:.1f}x base), momentum z {last['mom3']:+.2f}, "
        f"run {last['runlen']:+.0f}\n"
        f"Not financial advice"
    )
    if args.dry_run:
        print("[dry-run] would send:\n" + msg)
    else:
        send_telegram(msg)


if __name__ == "__main__":
    main()
