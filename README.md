# BTC/USD hourly inflection analysis

Do certain UTC clock hours carry a higher probability of a BTC/USD price
**inflection** (reversal or breakout)? And is any such pattern purely
clock-driven, or is it mediated by volume?

Full write-up with results, charts, and a review of prior published research:
**[REPORT.md](REPORT.md)**

## Layout

```
scripts/fetch_data.py   pull hourly BTC/USD OHLCV from Bitstamp (no API key)
scripts/analysis.py     event detection + hourly statistics + clock-vs-volume tests
data/btcusd_1h.csv      74,863 hourly candles, 2018-01-01 .. 2026-07-17 (UTC)
results/                figures, per-hour tables, summary.json
REPORT.md               findings + literature review
```

## Reproduce

```bash
pip install -r requirements.txt
python scripts/fetch_data.py --start 2018-01-01 --out data/btcusd_1h.csv
python scripts/analysis.py --data data/btcusd_1h.csv --outdir results
```

## Event definitions (hourly bars, UTC)

- **Reversal** — a confirmed local extremum of closes over a ±3-bar window,
  where both the move into and out of the extremum exceed 0.75× the trailing
  30-day standard deviation of hourly log returns (volatility-adaptive noise
  filter). Note the ±3-bar confirmation uses future bars, so this is a
  descriptive label, not a real-time signal.
- **Breakout** — the close exceeds the highest high (or lowest low) of the
  prior 24 bars (Donchian-24, i.e. a new daily-range extreme).

## Headline results

- Both event types cluster strongly by UTC hour (chi-square vs uniform:
  p ≈ 5e-28 for reversals, p ≈ 2e-72 for breakouts). Peak: 13:00–17:00 UTC
  (US session); trough: 02:00–08:00 UTC (Asian session); plus a local spike
  at exactly 00:00 UTC.
- The pattern is mostly a **liquidity/volume phenomenon** — contemporaneous
  volume explains 10–20× more deviance than clock hour — but a significant
  clock-hour effect survives volume control, and using only volume known
  *before* the bar, hour and volume carry comparable independent information.

See [REPORT.md](REPORT.md) for the full statistics, caveats, and references.
