#!/usr/bin/env python3
"""
FxBreezy Autonomous Market Analyst
----------------------------------
Runs the user's trading framework end-to-end, with no human in the loop:

  1. For every watchlist instrument, pull Daily / 4H / 1H / 30M candles (Yahoo, free).
  2. Establish higher-timeframe bias: Daily AND 4H must agree (trend or skip).
  3. Hunt a trend-CONTINUATION entry on 1H / 30M: pullback into value + a
     confirmation candle (engulfing / pin) + momentum, in the HTF direction.
  4. Size the trade: structure/ATR stop, TP1 + TP2, require R:R >= 2 (prefer 3+).
  5. Score confidence 0-100. Only setups >= 80% are sent.
  6. Push the alert to Telegram in the fixed format, de-duplicated so each
     setup fires once.

Credentials come from env vars (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) for
GitHub Actions, or from config.json for local runs. The token is never stored
in this file.

Usage:
  python scanner.py            # one full scan of the watchlist
  python scanner.py --ping     # send a "scanner online" test message
  python scanner.py --symbols EURUSD,XAUUSD   # scan a subset (debug)
  python scanner.py --dry      # analyse + print, never send to Telegram
"""
from __future__ import annotations
import os, sys, json, time, argparse, datetime as dt
import urllib.request, urllib.parse
import pandas as pd
import numpy as np

try:                                  # keep emoji-safe on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE   = os.path.dirname(os.path.abspath(__file__))
STATE  = os.path.join(BASE, "state.json")
TRADES = os.path.join(BASE, "trades.json")
BIASES = os.path.join(BASE, "bias_state.json")
ALERTS = os.path.join(BASE, "price_alerts.json")
CONF_THRESHOLD = 80          # only alert at or above this confidence
RR_MIN         = 2.0         # hard minimum risk:reward to TP2

# ── WATCHLIST ──────────────────────────────────────────────────────────────
# display name -> (yahoo ticker, pip size). Pip size only affects the "pips"
# shown in logs; prices are formatted to the instrument's natural precision.
WATCHLIST = {
    "BTCUSD":  ("BTC-USD",  1.0),
    "EURUSD":  ("EURUSD=X", 0.0001),
    "GBPCAD":  ("GBPCAD=X", 0.0001),
    "USDJPY":  ("USDJPY=X", 0.01),
    "AUDCAD":  ("AUDCAD=X", 0.0001),
    "EURAUD":  ("EURAUD=X", 0.0001),
    "NZDJPY":  ("NZDJPY=X", 0.01),
    "GBPAUD":  ("GBPAUD=X", 0.0001),
    "XAUUSD":  ("GC=F",     0.1),
    "USDCAD":  ("USDCAD=X", 0.0001),
    "NZDCAD":  ("NZDCAD=X", 0.0001),
    "GBPNZD":  ("GBPNZD=X", 0.0001),
    "NZDUSD":  ("NZDUSD=X", 0.0001),
    "GBPUSD":  ("GBPUSD=X", 0.0001),
    "USDCHF":  ("USDCHF=X", 0.0001),
    "GBPCHF":  ("GBPCHF=X", 0.0001),
    "GBPJPY":  ("GBPJPY=X", 0.01),
    "AUDUSD":  ("AUDUSD=X", 0.0001),
    "AUDJPY":  ("AUDJPY=X", 0.01),
    "EURJPY":  ("EURJPY=X", 0.01),
}

# ── small utils ─────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}", flush=True)

def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def creds():
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        cfg = load_json(os.path.join(BASE, "config.json"), {})
        tok = tok or cfg.get("telegram_bot_token")
        chat = chat or cfg.get("telegram_chat_id")
    return tok, chat

def send_telegram(text: str) -> bool:
    tok, chat = creds()
    if not (tok and chat):
        log("NO CREDS: set TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (or config.json)")
        return False
    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    payload = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=payload), timeout=20) as r:
            return json.loads(r.read().decode()).get("ok", False)
    except Exception as e:
        log(f"telegram send error: {e}")
        return False

# ── data layer (Yahoo Finance chart API — free, no key) ─────────────────────
def fetch(ticker: str, interval: str, rng: str) -> pd.DataFrame | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}"
    url += f"?interval={interval}&range={rng}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            j = json.loads(r.read().decode())
        res = j["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame(
            {"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
            index=pd.to_datetime(ts, unit="s", utc=True),
        ).dropna()
        return df if len(df) else None
    except Exception as e:
        log(f"fetch error {ticker} {interval}/{rng}: {e}")
        return None

def to_h4(df1h: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return df1h.resample("4h", label="right", closed="right").agg(agg).dropna()

def get_frames(ticker: str):
    d1 = fetch(ticker, "1d", "2y")
    h1 = fetch(ticker, "60m", "60d")
    m30 = fetch(ticker, "30m", "45d")
    if d1 is None or h1 is None or m30 is None:
        return None
    h4 = to_h4(h1)
    if len(d1) < 60 or len(h4) < 60 or len(h1) < 60 or len(m30) < 60:
        return None
    return d1, h4, h1, m30

# ── indicators ──────────────────────────────────────────────────────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def swings(df: pd.DataFrame, k: int = 5):
    """Return (swing_highs, swing_lows) as lists of (positional_index, price)."""
    h = df["high"].values; l = df["low"].values
    ph, pl = [], []
    for i in range(k, len(df) - k):
        if h[i] == max(h[i - k:i + k + 1]):
            ph.append((i, float(h[i])))
        if l[i] == min(l[i - k:i + k + 1]):
            pl.append((i, float(l[i])))
    return ph, pl

def structure_bias(df: pd.DataFrame, k: int = 5) -> str:
    """bull = HH+HL, bear = LH+LL, else none (last two swings each side)."""
    ph, pl = swings(df, k)
    if len(ph) < 2 or len(pl) < 2:
        return "none"
    if ph[-1][1] > ph[-2][1] and pl[-1][1] > pl[-2][1]:
        return "bull"
    if ph[-1][1] < ph[-2][1] and pl[-1][1] < pl[-2][1]:
        return "bear"
    return "none"

def trend_bias(df: pd.DataFrame) -> str:
    """bull / bear / none from EMA structure + slope + price location."""
    c = df["close"]
    slow_n = min(200, max(50, len(c) // 2))
    ef, es = ema(c, 50), ema(c, slow_n)
    last = c.iloc[-1]
    up = ef.iloc[-1] > es.iloc[-1] and last > ef.iloc[-1] and ef.iloc[-1] > ef.iloc[-4]
    dn = ef.iloc[-1] < es.iloc[-1] and last < ef.iloc[-1] and ef.iloc[-1] < ef.iloc[-4]
    return "bull" if up else "bear" if dn else "none"

def clean_structure(df: pd.DataFrame, direction: str, k: int = 3, look: int = 40) -> bool:
    """Confirm HH/HL (bull) or LH/LL (bear) from recent pivots."""
    seg = df.iloc[-look:]
    highs, lows = seg["high"].values, seg["low"].values
    ph, pl = [], []
    for i in range(k, len(seg) - k):
        if highs[i] == max(highs[i - k:i + k + 1]):
            ph.append(highs[i])
        if lows[i] == min(lows[i - k:i + k + 1]):
            pl.append(lows[i])
    if len(ph) < 2 or len(pl) < 2:
        return False
    if direction == "BUY":
        return ph[-1] > ph[-2] and pl[-1] > pl[-2]
    return ph[-1] < ph[-2] and pl[-1] < pl[-2]

# ── candle / entry logic on the trigger timeframe ───────────────────────────
def candle_signal(df: pd.DataFrame, direction: str):
    """
    Inspect the last CLOSED candle for a trend-continuation entry.
    Returns dict(entry, sl, kind, pullback, momentum, deep) or None.
    """
    c = df["close"]
    e20, e50 = ema(c, 20), ema(c, 50)
    a = atr(df, 14)
    if len(df) < 55 or pd.isna(a.iloc[-2]):
        return None

    # use the last fully-closed candle (-2); -1 may still be forming
    o, h, l, cl = (df["open"].iloc[-2], df["high"].iloc[-2],
                   df["low"].iloc[-2], df["close"].iloc[-2])
    po, pc = df["open"].iloc[-3], df["close"].iloc[-3]
    prng = df["high"].iloc[-3] - df["low"].iloc[-3]
    body = abs(cl - o)
    uw, lw = h - max(cl, o), min(cl, o) - l
    av = a.iloc[-2]
    ema20, ema50 = e20.iloc[-2], e50.iloc[-2]

    # pullback: the recent swing tagged the value zone (ema20..ema50) within ~0.6 ATR
    recent_low = df["low"].iloc[-6:-1].min()
    recent_high = df["high"].iloc[-6:-1].max()

    if direction == "BUY":
        engulf = cl > o and pc < po and body > prng * 0.8 and cl > df["high"].iloc[-3]
        pin = body > 0 and lw >= body * 2.0 and uw <= body * 0.4
        if not (engulf or pin):
            return None
        pullback = recent_low <= ema20 + 0.6 * av and cl > ema50      # dipped to value, trend intact
        deep = recent_low <= ema50 + 0.3 * av
        momentum = cl > ema20
        if not (pullback and momentum):
            return None
        entry = cl
        sl = min(recent_low, l) - 0.15 * av
        return dict(entry=entry, sl=sl, kind="Engulfing" if engulf else "Pin Bar",
                    pullback=True, deep=deep, momentum=momentum)
    else:
        engulf = cl < o and pc > po and body > prng * 0.8 and cl < df["low"].iloc[-3]
        pin = body > 0 and uw >= body * 2.0 and lw <= body * 0.4
        if not (engulf or pin):
            return None
        pullback = recent_high >= ema20 - 0.6 * av and cl < ema50
        deep = recent_high >= ema50 - 0.3 * av
        momentum = cl < ema20
        if not (pullback and momentum):
            return None
        entry = cl
        sl = max(recent_high, h) + 0.15 * av
        return dict(entry=entry, sl=sl, kind="Engulfing" if engulf else "Pin Bar",
                    pullback=True, deep=deep, momentum=momentum)

# ── session helper ──────────────────────────────────────────────────────────
def session_label() -> str | None:
    hr = dt.datetime.now(dt.timezone.utc).hour
    london = 7 <= hr < 16
    ny = 12 <= hr < 21
    if london and ny:
        return "London / New York"
    if london:
        return "London"
    if ny:
        return "New York"
    return None

# ── scoring ─────────────────────────────────────────────────────────────────
def score_setup(sig, struct_ok, rr2, in_session) -> int:
    s = 45                                   # HTF (D1+H4) aligned + valid trigger
    s += 18 if sig["kind"] == "Engulfing" else 10
    s += 12                                   # pullback into value (required)
    s += 6 if sig["deep"] else 0             # deeper retrace to EMA50
    s += 8 if sig["momentum"] else 0
    s += 12 if struct_ok else 0
    s += 5 if in_session else 0
    s += 7 if rr2 >= 3 else 3                 # rr2 always >= RR_MIN here
    return min(100, s)

# ── formatting ──────────────────────────────────────────────────────────────
def fmt_price(x: float, pip: float) -> str:
    digits = 0 if x >= 1000 else (2 if pip >= 0.1 else (3 if pip >= 0.01 else 5))
    return f"{x:.{digits}f}"

SEP = "━" * 18

def sym_disp(name: str) -> str:
    return f"{name[:3]}/{name[3:]}" if len(name) == 6 else name

def pbar(pct: float, n: int = 10) -> str:
    pct = max(0.0, min(100.0, pct))
    fill = int(round(pct / 100 * n))
    return "█" * fill + "░" * (n - fill)

def fmt_dt(ts) -> str:
    try:
        d = pd.Timestamp(ts).to_pydatetime()
    except Exception:
        d = dt.datetime.now(dt.timezone.utc)
    return d.strftime("%d %b %Y | %H:%M")

def fmt_dur(a, b) -> str:
    try:
        secs = (pd.Timestamp(b) - pd.Timestamp(a)).total_seconds()
        m = int(secs // 60)
        return f"{m // 60}h {m % 60}m"
    except Exception:
        return "—"

def build_alert(name, tf, direction, sig, tp1, tp2, rr2, conf, pip, bias_txt,
                invalidation) -> str:
    p = lambda v: fmt_price(v, pip)
    head = "📈 BUY SIGNAL" if direction == "BUY" else "📉 SELL SIGNAL"
    pa = "Bullish PA" if direction == "BUY" else "Bearish PA"
    score_arrow = "⬆️" if direction == "BUY" else "⬇️"
    sl = session_label()
    sess = f"{sl} Session Open" if sl else "Outside main sessions"
    return (
        f"{head} — {name}\n"
        f"📍 Entry: {p(sig['entry'])}\n"
        f"🛑 Stop Loss: {p(sig['sl'])}\n"
        f"🎯 TP1: {p(tp1)}\n"
        f"🎯 TP2: {p(tp2)}\n"
        f"📊 R:R — 1:{rr2:g}\n"
        f"⏰ {sess}\n"
        f"✅ Confirmed — High Volume + {pa}\n"
        f"{score_arrow} Setup Score: {conf}%  (rule-based)"
    )

# ── per-symbol analysis ─────────────────────────────────────────────────────
def analyse_frames(name: str, frames, pip: float):
    d1, h4, h1, m30 = frames
    bias_d, bias_h4 = trend_bias(d1), trend_bias(h4)
    if bias_d == "none" or bias_d != bias_h4:        # HTF must agree
        return None
    direction = "BUY" if bias_d == "bull" else "SELL"

    for tf_name, df in (("30M", m30), ("1H", h1)):
        sig = candle_signal(df, direction)
        if not sig:
            continue
        # freshness: only act on a just-closed candle (also skips stale/weekend data)
        bar_min = 30 if tf_name == "30M" else 60
        age_min = (pd.Timestamp.now(tz="UTC") - df.index[-2]).total_seconds() / 60
        if age_min > bar_min * 2.5:
            continue
        risk = abs(sig["entry"] - sig["sl"])
        if risk <= 0:
            continue
        if direction == "BUY":
            tp1 = sig["entry"] + risk * 1.0
            tp2 = sig["entry"] + risk * 2.0
        else:
            tp1 = sig["entry"] - risk * 1.0
            tp2 = sig["entry"] - risk * 2.0
        rr2 = abs(tp2 - sig["entry"]) / risk
        if rr2 < RR_MIN:
            continue
        struct_ok = clean_structure(df, direction)
        in_sess = session_label() is not None
        conf = score_setup(sig, struct_ok, rr2, in_sess)
        if conf < CONF_THRESHOLD:
            continue
        bias_txt = (f"D1 {bias_d.upper()} + H4 {bias_h4.upper()} aligned"
                    f"{' · clean structure' if struct_ok else ''}")
        invalid = ("Close beyond stop / loss of " +
                   ("HL" if direction == "BUY" else "LH") + " structure")
        sig_time = str(df.index[-2])
        return dict(name=name, tf=tf_name, direction=direction, sig=sig,
                    tp1=tp1, tp2=tp2, rr2=rr2, conf=conf, pip=pip,
                    bias_txt=bias_txt, invalidation=invalid, sig_time=sig_time)
    return None

# ── weekend gap analysis ────────────────────────────────────────────────────
def weekend_gap(h1: pd.DataFrame):
    """Find the weekend break (largest inter-bar time gap) and measure it."""
    idx = h1.index
    if len(idx) < 3:
        return None
    diffs = idx.to_series().diff()
    pos = int(np.nanargmax(diffs.values[1:])) + 1     # position of the biggest gap
    if diffs.iloc[pos].total_seconds() / 3600 < 12:   # not a weekend-sized break
        return None
    cb = float(h1["close"].iloc[pos - 1])
    oa = float(h1["open"].iloc[pos])
    if cb == 0:
        return None
    return dict(close_before=cb, open_after=oa, gap=oa - cb,
                pct=(oa - cb) / cb * 100, when=idx[pos])

def build_gap_report(rows) -> str:
    lines = ["🗓 WEEKEND GAP REPORT", f"{len(rows)} instrument(s) gapped at the open:\n"]
    for name, g, bias, pip in rows:
        up = g["gap"] > 0
        arrow = "🔼 GAP UP" if up else "🔽 GAP DOWN"
        if (bias == "bull" and up) or (bias == "bear" and not up):
            ctx = "supports the D1 trend → watch for continuation confirmation"
        elif bias in ("bull", "bear"):
            ctx = "counter-trend gap → statistically likely to fill"
        else:
            ctx = "no clear D1 trend → treat as range, expect a fill"
        lines.append(f"{arrow}  {name}  {g['pct']:+.2f}%")
        lines.append(f"   {fmt_price(g['close_before'], pip)} → {fmt_price(g['open_after'], pip)}  ·  D1 {bias.upper()}")
        lines.append(f"   {ctx}")
    lines.append("\n⚠️ Don't trade the gap blind — wait for confirmation before entering.")
    return "\n".join(lines)

def gap_scan(symbols=None, dry=False):
    state = load_json(STATE, {})
    targets = symbols or [n for n in WATCHLIST if n != "BTCUSD"]   # crypto trades 24/7
    rows, newest = [], None
    for name in targets:
        if name not in WATCHLIST:
            continue
        ticker, pip = WATCHLIST[name]
        h1 = fetch(ticker, "60m", "7d")
        if h1 is None or len(h1) < 10:
            continue
        g = weekend_gap(h1)
        if not g:
            continue
        age_h = (pd.Timestamp.now(tz="UTC") - g["when"]).total_seconds() / 3600
        if age_h > 36 or abs(g["pct"]) < 0.15:        # only fresh + meaningful gaps
            continue
        d1 = fetch(ticker, "1d", "2y")
        bias = trend_bias(d1) if d1 is not None and len(d1) > 60 else "none"
        rows.append((name, g, bias, pip))
        newest = g["when"] if newest is None or g["when"] > newest else newest
        time.sleep(0.3)
    if not rows:
        log("gap scan: no significant weekend gaps")
        return
    rows.sort(key=lambda r: abs(r[1]["pct"]), reverse=True)
    key = f"gapreport:{str(newest)[:10]}"
    if state.get(key) and not dry:
        log("gap report already sent this weekend")
        return
    msg = build_gap_report(rows)
    if dry:
        print("\n" + msg + "\n")
    elif send_telegram(msg):
        state[key] = {"sent_at": dt.datetime.now(dt.timezone.utc).isoformat(), "count": len(rows)}
        save_json(STATE, state)
    log(f"gap scan complete: {len(rows)} gap(s) reported")

# ── trade tracking (open / monitor TP-SL / portfolio) ──────────────────────
# Model: 1 unit risking 1R to the original SL, target = TP2 (1:2).
#   SL hit before TP1  -> LOSS (-1R)
#   TP1 hit            -> milestone alert, stop moves to breakeven (entry)
#   TP2 hit            -> WIN (+2R)
#   stopped at entry after TP1 -> BREAKEVEN (0R)
def notify(text: str, dry: bool = False) -> bool:
    if dry:
        print("\n" + text + "\n")
        return True
    return send_telegram(text)

def record_trade(setup: dict) -> None:
    trades = load_json(TRADES, [])
    tid = f"{setup['name']}:{setup['direction']}:{setup['tf']}:{setup['sig_time']}"
    if any(t["id"] == tid for t in trades):
        return
    s = setup["sig"]
    trades.append({
        "id": tid, "symbol": setup["name"], "ticker": WATCHLIST[setup["name"]][0],
        "pip": setup["pip"], "tf": setup["tf"], "direction": setup["direction"],
        "entry": s["entry"], "sl": s["sl"], "tp1": setup["tp1"], "tp2": setup["tp2"],
        "conf": setup["conf"], "opened_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sig_time": setup["sig_time"], "last_checked": setup["sig_time"],
        "status": "open", "stop": s["sl"], "tp1_hit": False,
    })
    save_json(TRADES, trades)

def _tp1_msg(t):
    return (
        f"🎯 FxBreezy UPDATE | {sym_disp(t['symbol'])}\n"
        f"TP1 REACHED ✅\n\n"
        f"Profit: +1R\n"
        f"Action: SL moved to BE 🔒\n"
        f"Status: RUNNING\n"
        f"Next Target: TP2 (+2R)\n"
        f"📊 Trade Protection: 100%"
    )

def _close_msg(t):
    s = sym_disp(t["symbol"]); d = t["direction"]; r = t["result"]
    if r == "win":
        return (
            f"🏆 FxBreezy RESULT | {s}\n"
            f"{d} CLOSED ✅\n\n"
            f"Outcome: WIN\n"
            f"Return: +2.0R\n"
            f"Trade: Entry → TP1 → TP2\n"
            f"Duration: {fmt_dur(t.get('opened_at'), t.get('closed_at'))}\n"
            f"#FxBreezy"
        )
    if r == "loss":
        return (
            f"🛑 FxBreezy RESULT | {s}\n"
            f"{d} CLOSED\n\n"
            f"Outcome: LOSS\n"
            f"Return: -1.0R\n"
            f"Reason: SL Hit\n"
            f"Risk Managed: 1R"
        )
    return (
        f"⚪ FxBreezy RESULT | {s}\n"
        f"Protected Exit\n\n"
        f"Outcome: BE\n"
        f"Return: 0R\n"
        f"Flow: Entry → TP1 → BE\n"
        f"Capital Protected 🔒"
    )

def monitor_trades(dry: bool = False) -> int:
    trades = load_json(TRADES, [])
    closes = 0
    for t in trades:
        if t["status"] not in ("open", "tp1"):
            continue
        df = fetch(t["ticker"], "30m", "7d")
        if df is None or df.empty:
            continue
        try:
            start = pd.Timestamp(t.get("last_checked") or t["sig_time"])
        except Exception:
            start = df.index[0]
        bars = df[df.index > start]
        buy = t["direction"] == "BUY"
        for ts, row in bars.iterrows():
            hi, lo = float(row["high"]), float(row["low"])
            stop, tp1, tp2 = t["stop"], t["tp1"], t["tp2"]
            hit_stop = lo <= stop if buy else hi >= stop
            hit_tp1  = hi >= tp1 if buy else lo <= tp1
            hit_tp2  = hi >= tp2 if buy else lo <= tp2
            if hit_stop:                                   # pessimistic: stop checked first
                be = abs(stop - t["entry"]) < t["entry"] * 1e-6
                t.update(status="closed", result="breakeven" if be else "loss",
                         r_multiple=0.0 if be else -1.0, exit_price=stop, closed_at=str(ts))
                notify(_close_msg(t), dry); closes += 1
                break
            if not t["tp1_hit"] and hit_tp1:
                t["tp1_hit"] = True; t["status"] = "tp1"; t["stop"] = t["entry"]
                notify(_tp1_msg(t), dry)
            if hit_tp2:
                t.update(status="closed", result="win", r_multiple=2.0,
                         exit_price=tp2, closed_at=str(ts))
                notify(_close_msg(t), dry); closes += 1
                break
        if bars.shape[0]:
            t["last_checked"] = str(bars.index[-1])
    save_json(TRADES, trades)
    if closes:
        notify(portfolio_summary(), dry)
    return closes

def portfolio_summary() -> str:
    trades = load_json(TRADES, [])
    closed = [t for t in trades if t["status"] == "closed"]
    opn = [t for t in trades if t["status"] in ("open", "tp1")]
    wins = [t for t in closed if t["result"] == "win"]
    loss = [t for t in closed if t["result"] == "loss"]
    be = [t for t in closed if t["result"] == "breakeven"]
    decisive = len(wins) + len(loss)
    wr = (len(wins) / decisive * 100) if decisive else 0
    net = sum(t.get("r_multiple", 0) for t in closed)
    active_risk = sum(0 if t["tp1_hit"] else 1 for t in opn)
    return (
        f"📊 FxBreezy PORTFOLIO\n\n"
        f"Closed Trades: {len(closed)}\n"
        f"✅ Wins: {len(wins)}\n"
        f"🛑 Losses: {len(loss)}\n"
        f"⚪ BE: {len(be)}\n\n"
        f"Win Rate: {wr:.0f}%\n"
        f"Net: {net:+.1f}R\n\n"
        f"Open Trades: {len(opn)}\n"
        f"Risk: {active_risk}R\n"
        f"System: 🟢 Healthy"
    )

def stats_dashboard() -> str:
    trades = load_json(TRADES, [])
    closed = [t for t in trades if t["status"] == "closed" and t.get("closed_at")]
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()

    def cdate(t):
        try:
            return pd.Timestamp(t["closed_at"]).to_pydatetime().astimezone(dt.timezone.utc).date()
        except Exception:
            return None

    td = [t for t in closed if cdate(t) == today]
    wk = [t for t in closed if cdate(t) and (today - cdate(t)).days < 7]
    mo = [t for t in closed if cdate(t) and cdate(t).year == today.year and cdate(t).month == today.month]

    def winrate(ts):
        w = sum(1 for t in ts if t["result"] == "win")
        l = sum(1 for t in ts if t["result"] == "loss")
        return (w / (w + l) * 100) if (w + l) else 0

    td_w = sum(1 for t in td if t["result"] == "win")
    td_l = sum(1 for t in td if t["result"] == "loss")
    td_pl = sum(t.get("r_multiple", 0) for t in td)
    mo_net = sum(t.get("r_multiple", 0) for t in mo)
    opn = [t for t in trades if t["status"] in ("open", "tp1")]
    return (
        "📊 FxBreezy Live Dashboard\n\n"
        f"Today:\nTrades: {len(td)}\nWins: {td_w}\nLosses: {td_l}\n\n"
        f"Today P/L:\n{td_pl:+.0f}R\n\n"
        f"This Week:\nWin Rate:\n{winrate(wk):.0f}%\n\n"
        f"Monthly:\n{mo_net:+.1f}R\n\n"
        f"Current:\n{len(opn)} Open Trades"
    )

def listen():
    """Long-poll Telegram and reply to /stats and /summary. Self-healing singleton."""
    tok, chat = creds()
    if not (tok and chat):
        log("listen: no creds")
        return
    # singleton guard: if another listener's heartbeat is fresh (<120s), don't double-poll
    lock = os.path.join(BASE, "listener.lock")
    try:
        if os.path.exists(lock) and (time.time() - os.path.getmtime(lock)) < 120:
            log("listener already running (fresh lock) — exiting duplicate")
            return
    except Exception:
        pass
    base = f"https://api.telegram.org/bot{tok}"
    off_file = os.path.join(BASE, "tg_offset.txt")
    try:
        offset = int(open(off_file).read().strip())
    except Exception:
        offset = 0
    log("listener started — commands: /stats /summary /help")
    while True:
        try:
            try:
                open(lock, "w").write(str(time.time()))      # heartbeat
            except Exception:
                pass
            url = f"{base}/getUpdates?timeout=50" + (f"&offset={offset}" if offset else "")
            with urllib.request.urlopen(url, timeout=70) as r:
                data = json.loads(r.read().decode())
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                try:
                    open(off_file, "w").write(str(offset))
                except Exception:
                    pass
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip().lower()
                cid = str((msg.get("chat") or {}).get("id"))
                if cid != str(chat):
                    continue
                if text.startswith("/stats"):
                    send_telegram(stats_dashboard())
                elif text.startswith("/summary"):
                    send_telegram(portfolio_summary())
                elif text.startswith(("/start", "/help")):
                    send_telegram("🤖 FxBreezy bot online.\nCommands:\n/stats — live dashboard\n/summary — full portfolio")
        except Exception as e:
            log(f"listen error: {e}")
            time.sleep(5)

# ── main scan ───────────────────────────────────────────────────────────────
def overall_bias(d1, h4) -> str:
    """Combined HTF bias: bull/bear only when Daily and 4H agree, else none."""
    bd, bh = trend_bias(d1), trend_bias(h4)
    return bd if (bd == bh and bd != "none") else "none"

def trend_change_msg(name, prev, new) -> str:
    if new == "bear":
        arrow = "🔻 BULLISH → BEARISH"
        ev = "Daily + 4H structure flipped to LH + LL; momentum bearish"
        plan = "Stop hunting BUYs. Wait for a bearish break-&-retest."
    else:
        arrow = "🔺 BEARISH → BULLISH"
        ev = "Daily + 4H structure flipped to HH + HL; momentum bullish"
        plan = "Stop hunting SELLs. Wait for a bullish break-&-retest."
    return (
        f"⚠️ TREND CHANGE | {sym_disp(name)}\n\n"
        f"{arrow}\n"
        f"{'━' * 12}\n"
        f"Evidence:\n{ev}\n\n"
        f"New Plan:\n{plan}\n\n"
        f"#FxBreezy"
    )

def price_alert_msg(name, level, direction, note, pip) -> str:
    d = "ABOVE" if direction == "above" else "BELOW"
    body = f"🔔 PRICE ALERT | {sym_disp(name)}\n\nPrice crossed {d} {fmt_price(level, pip)}"
    if note:
        body += f"\n{note}"
    return body + "\n\n#FxBreezy"

def run_scan(symbols=None, dry=False):
    monitor_trades(dry)                       # update open trades & fire TP/SL alerts first
    state = load_json(STATE, {})
    biases = load_json(BIASES, {})
    palerts = load_json(ALERTS, {})
    targets = symbols or list(WATCHLIST.keys())
    found, sent, flips = 0, 0, 0
    for name in targets:
        if name not in WATCHLIST:
            log(f"skip unknown symbol {name}")
            continue
        ticker, pip = WATCHLIST[name]
        frames = get_frames(ticker)
        if not frames:
            log(f"{name}: data unavailable")
            continue
        d1, h4, h1, m30 = frames

        # ── trend-change detection: alert when the agreed Daily+4H bias flips ──
        ob = overall_bias(d1, h4)
        if ob != "none":
            prev = biases.get(name)
            if prev in ("bull", "bear") and prev != ob:
                log(f"{name}: TREND CHANGE {prev}->{ob}")
                flips += 1
                msg = trend_change_msg(name, prev, ob)
                if dry:
                    print("\n" + msg + "\n")
                else:
                    send_telegram(msg)
            biases[name] = ob
            save_json(BIASES, biases)

        # ── price-level alerts (key levels to watch) ──
        watch = palerts.get(name)
        if watch:
            price = float(h1["close"].iloc[-1])
            keep = []
            for al in watch:
                hit = ((al["dir"] == "above" and price >= al["level"]) or
                       (al["dir"] == "below" and price <= al["level"]))
                if hit:
                    log(f"{name}: PRICE ALERT {al['dir']} {al['level']}")
                    m = price_alert_msg(name, al["level"], al["dir"], al.get("note", ""), pip)
                    print("\n" + m + "\n") if dry else send_telegram(m)
                else:
                    keep.append(al)
            if len(keep) != len(watch):
                palerts[name] = keep
                save_json(ALERTS, palerts)

        # ── break-and-retest setup ──
        try:
            setup = analyse_frames(name, frames, pip)
        except Exception as e:
            log(f"analyse error {name}: {e}")
            continue
        if not setup:
            log(f"{name}: no qualifying setup")
            continue
        found += 1
        key = f"{setup['name']}:{setup['direction']}:{setup['tf']}:{setup['sig_time']}"
        if state.get(key):
            log(f"{name}: setup already alerted")
            continue
        msg = build_alert(setup["name"], setup["tf"], setup["direction"], setup["sig"],
                          setup["tp1"], setup["tp2"], setup["rr2"], setup["conf"],
                          setup["pip"], setup["bias_txt"], setup["invalidation"])
        log(f"{name}: SIGNAL {setup['direction']} {setup['tf']} score={setup['conf']}%")
        if dry:
            print("\n" + msg + "\n")
        else:
            if send_telegram(msg):
                state[key] = {"sent_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "score": setup["conf"]}
                save_json(STATE, state)
                record_trade(setup)            # log it as an open trade to track TP/SL
                sent += 1
        time.sleep(0.4)
    log(f"scan complete: {len(targets)} scanned, {found} setups, {sent} alerts, {flips} trend-changes")
    return found, sent

# ── entry point ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ping", action="store_true", help="send a test message and exit")
    ap.add_argument("--dry", action="store_true", help="analyse + print, never send")
    ap.add_argument("--gap", action="store_true", help="run weekend-gap analysis instead of a scan")
    ap.add_argument("--monitor", action="store_true", help="only monitor open trades for TP/SL")
    ap.add_argument("--summary", action="store_true", help="send the portfolio summary and exit")
    ap.add_argument("--stats", action="store_true", help="send the live /stats dashboard and exit")
    ap.add_argument("--listen", action="store_true", help="long-poll Telegram for /stats /summary (runs forever)")
    ap.add_argument("--symbols", help="comma-separated subset, e.g. EURUSD,XAUUSD")
    args = ap.parse_args()

    if args.ping:
        ok = send_telegram("🤖 FxBreezy analyst online — scanner reachable and authorised.")
        log(f"ping sent ok={ok}")
        return
    if args.listen:
        listen(); return
    if args.stats:
        notify(stats_dashboard(), args.dry); return
    if args.summary:
        notify(portfolio_summary(), args.dry); return
    if args.monitor:
        n = monitor_trades(args.dry); log(f"monitor: {n} trade(s) closed"); return
    syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
    if args.gap:
        gap_scan(symbols=syms, dry=args.dry)
    else:
        run_scan(symbols=syms, dry=args.dry)

if __name__ == "__main__":
    main()
