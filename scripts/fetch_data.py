"""Fetch hourly BTC/USD OHLCV candles from the Bitstamp public API.

Bitstamp trades real BTC/USD (fiat), has data back to 2011, and its public
OHLC endpoint needs no API key. We page through it 1000 candles at a time.

Usage:
    python scripts/fetch_data.py [--start 2018-01-01] [--out data/btcusd_1h.csv]
"""

import argparse
import csv
import datetime as dt
import time
import urllib.parse
import urllib.request

API = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
STEP = 3600  # 1h candles
PAGE = 1000  # max candles per request


def fetch_page(start_ts: int) -> list[dict]:
    params = urllib.parse.urlencode(
        {"step": STEP, "limit": PAGE, "start": start_ts}
    )
    req = urllib.request.Request(f"{API}?{params}", headers={"User-Agent": "btc-inflection-research/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                import json

                payload = json.load(resp)
            return payload["data"]["ohlc"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", default="data/btcusd_1h.csv")
    args = ap.parse_args()

    start_ts = int(
        dt.datetime.strptime(args.start, "%Y-%m-%d")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )
    now_ts = int(time.time())

    rows: list[dict] = []
    cursor = start_ts
    while cursor < now_ts:
        page = fetch_page(cursor)
        if not page:
            break
        rows.extend(page)
        last_ts = int(page[-1]["timestamp"])
        if last_ts <= cursor:  # no forward progress -> reached the end
            break
        cursor = last_ts + STEP
        print(
            f"fetched through {dt.datetime.fromtimestamp(last_ts, dt.timezone.utc):%Y-%m-%d %H:%M} "
            f"({len(rows)} rows)",
            flush=True,
        )
        time.sleep(0.15)  # stay well under rate limits

    # de-dup on timestamp, keep chronological order
    seen: dict[int, dict] = {}
    for r in rows:
        seen[int(r["timestamp"])] = r
    ordered = [seen[k] for k in sorted(seen)]

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for r in ordered:
            w.writerow(
                [r["timestamp"], r["open"], r["high"], r["low"], r["close"], r["volume"]]
            )
    print(f"wrote {len(ordered)} candles to {args.out}")


if __name__ == "__main__":
    main()
