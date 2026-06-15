# FxBreezy Autonomous Market Analyst

Scans your 20-instrument watchlist on the user's framework and sends **only**
high-probability (≥80% confidence) trend-continuation signals to Telegram.

- **Bias:** Daily **and** 4H must agree (else the pair is skipped).
- **Entry:** 1H / 30M pullback into value + confirmation candle (engulfing/pin)
  + momentum, in the HTF direction.
- **Risk:** structure/ATR stop, TP1 (1R) + TP2 (2R), **R:R ≥ 1:2** enforced.
- **Selectivity:** confidence scored 0–100; only ≥80 is sent. De-duplicated so
  each setup fires once. Freshness-gated so it never alerts on stale candles.
- **Data:** Yahoo Finance chart API — free, no key.

## Run it 24/7 for free (GitHub Actions — recommended)

1. Push this folder **and** `.github/workflows/fxbreezy-scan.yml` to a GitHub repo.
2. Repo → **Settings → Secrets and variables → Actions → New repository secret**:
   - `TELEGRAM_BOT_TOKEN`  → your BotFather token
   - `TELEGRAM_CHAT_ID`    → `5065818386`
3. Repo → **Actions** tab → enable workflows. It now runs **every 30 minutes**.
   Use **Run workflow** to fire one manually.

Notes: cron is UTC and can be delayed a few minutes under load; GitHub pauses
scheduled workflows after 60 days of repo inactivity (any push re-arms them).
The token lives only in the encrypted secret — never in the code.

## Run it locally (testing)

```powershell
cd fxbreezy_bot
pip install -r requirements.txt
copy config.example.json config.json   # then paste token + chat id
python scanner.py --ping               # sends a test message
python scanner.py --dry                # analyse + print, never sends
python scanner.py                      # real scan -> Telegram on a valid setup
python scanner.py --symbols EURUSD,XAUUSD   # scan a subset
```

## Tuning

In `scanner.py`: `CONF_THRESHOLD` (default 80), `RR_MIN` (default 2.0), and the
`WATCHLIST` dict (display name → Yahoo ticker). Confidence weighting lives in
`score_setup()`.
