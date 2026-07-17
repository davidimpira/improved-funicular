"""Live scorer: fetch recent candles, score reversal probability, alert.

Designed to run from cron (e.g. GitHub Actions, a few minutes past each
hour). Fetches the last ~40 days of hourly BTC/USD candles from Bitstamp,
rebuilds the causal features, scores the last CLOSED bar with the saved
logistic model, and sends a Telegram message when the probability crosses
the alert threshold.

Alert rule (stateless, no DB needed):
  alert iff p[last] >= threshold AND p[previous] < threshold
i.e. only on upward threshold crossings, so a stretch of consecutive hot
hours produces one alert, not one per hour.

Environment:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   if unset, prints instead of sending
  ALERT_MIN_LIFT   alert threshold as a multiple of base rate (default 2.5;
                   see results/live_signal_backtest.md before changing)

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
    min_lift = float(os.environ.get("ALERT_MIN_LIFT", "2.5"))
    threshold = base * min_lift

    feats = build_features(fetch_recent())
    scored = feats.dropna(subset=FEATURES)
    if len(scored) < 2:
        print("not enough complete bars with features; aborting")
        sys.exit(1)

    last, prev = scored.iloc[-1], scored.iloc[-2]
    p_last = predict_proba(last, model)
    p_prev = predict_proba(prev, model)
    lift = p_last / base

    bar_ts = scored.index[-1]
    print(f"bar {bar_ts:%Y-%m-%d %H:%M} UTC | close {last['close']:,.0f} | "
          f"P(reversal<=3 bars) {p_last:.1%} (lift {lift:.2f}x, "
          f"prev {p_prev:.1%}, threshold {threshold:.1%})")

    crossing = p_last >= threshold and p_prev < threshold
    if not crossing:
        print("no alert (below threshold or already alerted on this stretch)")
        return

    msg = (
        f"BTC/USD reversal watch — {bar_ts:%H:%M} UTC bar\n"
        f"P(confirmed reversal within 3 bars): {p_last:.1%} "
        f"({lift:.1f}x the {base:.1%} base rate)\n"
        f"Price: {last['close']:,.0f}\n"
        f"Drivers: volume {math.exp(last['vol_surprise']):.2f}x this hour's norm, "
        f"3-bar momentum z {last['mom3']:+.2f}, run length {last['runlen']:+.0f}\n"
        f"Model: {model['target']} | not financial advice"
    )
    if args.dry_run:
        print("[dry-run] would send:\n" + msg)
    else:
        send_telegram(msg)


if __name__ == "__main__":
    main()
