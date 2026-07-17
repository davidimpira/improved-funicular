# BTC/USD hourly inflection analysis

Do certain UTC clock hours carry a higher probability of a BTC/USD price
**inflection** (reversal or breakout)? And is any such pattern purely
clock-driven, or is it mediated by volume?

Full write-up with results, charts, and a review of prior published research:
**[REPORT.md](REPORT.md)**

## Layout

```
scripts/fetch_data.py            pull hourly BTC/USD OHLCV from Bitstamp (no API key)
scripts/analysis.py              event detection + hourly statistics + clock-vs-volume tests
scripts/live_features.py         causal features shared by training and live scoring
scripts/train_reversal_model.py  walk-forward validation + final model fit
scripts/live_score.py            hourly live scorer + Telegram alerting
.github/workflows/reversal-alert.yml  hourly cron that runs the live scorer
models/reversal_model.json       shipped logistic model (coefs + base rate)
data/btcusd_1h.csv               74,863 hourly candles, 2018-01-01 .. 2026-07-17 (UTC)
results/                         figures, tables, summary.json, live_signal_backtest.md
REPORT.md                        findings + literature review
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

## Live reversal alerts

A live tool built on the findings above: every hour it scores
**P(confirmed reversal within the next 3 bars)** for the just-closed BTC/USD
bar using only causally available features (UTC hour, volume surprise vs that
hour's own norm, recent volume build-up, momentum, distance from the 24h range,
run length) and sends a Telegram message when the probability crosses the
alert threshold.

Walk-forward validation (2020–2026, each year scored by a model trained only
on prior years — [results/live_signal_backtest.md](results/live_signal_backtest.md)):
AUC 0.694, stable per-year (0.65–0.73), monotonic calibration. At the default
threshold (2.5× the 11% base rate) it alerts ~2.8×/week and ~38% of alerts are
followed by a confirmed reversal within 3 hours (3.5× lift).

### Setup

1. Create a Telegram bot: message [@BotFather](https://t.me/botfather),
   `/newbot`, copy the token.
2. Get your chat id: message your new bot once, then open
   `https://api.telegram.org/bot<TOKEN>/getUpdates` and read
   `message.chat.id`.
3. In the GitHub repo: Settings → Secrets and variables → Actions → add
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
4. The [`reversal-alert`](.github/workflows/reversal-alert.yml) workflow runs
   at :06 past every hour (enable workflows in the Actions tab if prompted).
   Trigger it manually via *Run workflow* to test; without secrets set it
   prints the message instead of sending.

Alerts fire only on upward threshold *crossings* (a stretch of consecutive
hot hours produces one alert). Tune sensitivity with the `ALERT_MIN_LIFT` env
in the workflow, using the trade-off table in the backtest report.

### Retrain / run locally

```bash
python scripts/train_reversal_model.py --data data/btcusd_1h.csv   # refit + backtest report
python scripts/live_score.py --dry-run                             # score latest bar now
```

Retrain occasionally (e.g. quarterly) after refreshing the dataset with
`fetch_data.py`. **Not financial advice; an elevated probability is a
statistical tendency, not a prediction.**
