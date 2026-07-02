#!/usr/bin/env python3
"""
FxBreezy Backtest Harness
-------------------------
Walk-forward simulation of the live scanner logic over the last ~60 days of
Yahoo intraday data (their free limit for 30m/60m bars). Reuses the exact
entry/exit rules from scanner.py so what we test is what actually runs.

No lookahead:
  - The signal candle is the last CLOSED bar (i-1) while bar i is "forming".
  - EMAs/ATR use adjust=False ewm, which is causal — precomputing on the full
    frame gives identical values to prefix computation.
  - Daily / 4H bias uses only closed HTF bars plus the running price at
    decision time (mirrors live, which includes the forming HTF candle).

Costs are NOT modelled (live alerts are mid-price too) — results are
comparative between variants, not broker-accurate.

Usage:
  python backtest.py                # run all variants on the full watchlist
  python backtest.py --symbols X,Y  # subset (debug)
"""
from __future__ import annotations
import sys, json, time, argparse
import datetime as dt
import pandas as pd
import numpy as np

from scanner import (WATCHLIST, fetch, to_h4, ema, atr,
                     clean_structure, score_setup, CONF_THRESHOLD,
                     COOLDOWN_HOURS, MAX_SIGNALS_PER_DAY)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

WARMUP_BARS = 100          # skip first bars of the entry frame (indicator warmup)


# ── point-in-time HTF bias ──────────────────────────────────────────────────
def bias_from_closes(closes: pd.Series) -> str:
    """trend_bias() but on a close series (it only ever reads closes)."""
    if len(closes) < 60:
        return "none"
    slow_n = min(200, max(50, len(closes) // 2))
    ef, es = ema(closes, 50), ema(closes, slow_n)
    last = closes.iloc[-1]
    up = ef.iloc[-1] > es.iloc[-1] and last > ef.iloc[-1] and ef.iloc[-1] > ef.iloc[-4]
    dn = ef.iloc[-1] < es.iloc[-1] and last < ef.iloc[-1] and ef.iloc[-1] < ef.iloc[-4]
    return "bull" if up else "bear" if dn else "none"


class HTFBias:
    """Caches D1+H4 agreement bias per (daily-cutoff, h4-cutoff) step."""

    def __init__(self, d1: pd.DataFrame, h4: pd.DataFrame):
        self.d1_closes = d1["close"]
        self.h4_closes = h4["close"]          # h4 index = bar END time (label=right)
        self._cache: dict = {}

    def at(self, t: pd.Timestamp, price_now: float) -> str:
        day = t.normalize()
        n_d1 = self.d1_closes.index.searchsorted(day)            # closed daily bars
        n_h4 = self.h4_closes.index.searchsorted(t, side="right")  # closed h4 bars
        key = (n_d1, n_h4)
        if key not in self._cache:
            d1c = pd.concat([self.d1_closes.iloc[:n_d1],
                             pd.Series([price_now])], ignore_index=True)
            h4c = pd.concat([self.h4_closes.iloc[:n_h4],
                             pd.Series([price_now])], ignore_index=True)
            bd, bh = bias_from_closes(d1c), bias_from_closes(h4c)
            self._cache[key] = bd if (bd == bh and bd != "none") else "none"
        # price_now varies within a cached step; bias is slow — acceptable.
        return self._cache[key]


# ── entry check at bar i (mirror of candle_signal, precomputed indicators) ──
def candle_signal_at(df, e20, e50, a, i, direction):
    """Bar i is 'forming'; decide off closed bars i-1 / i-2 (live -2 / -3)."""
    if i < 55 or pd.isna(a.iloc[i - 1]):
        return None
    o, h, l, cl = (df["open"].iloc[i - 1], df["high"].iloc[i - 1],
                   df["low"].iloc[i - 1], df["close"].iloc[i - 1])
    po, pc = df["open"].iloc[i - 2], df["close"].iloc[i - 2]
    prng = df["high"].iloc[i - 2] - df["low"].iloc[i - 2]
    body = abs(cl - o)
    uw, lw = h - max(cl, o), min(cl, o) - l
    av = a.iloc[i - 1]
    ema20, ema50 = e20.iloc[i - 1], e50.iloc[i - 1]
    recent_low = df["low"].iloc[max(0, i - 5):i].min()
    recent_high = df["high"].iloc[max(0, i - 5):i].max()

    if direction == "BUY":
        engulf = cl > o and pc < po and body > prng * 0.8 and cl > df["high"].iloc[i - 2]
        pin = body > 0 and lw >= body * 2.0 and uw <= body * 0.4
        if not (engulf or pin):
            return None
        pullback = recent_low <= ema20 + 0.6 * av and cl > ema50
        deep = recent_low <= ema50 + 0.3 * av
        momentum = cl > ema20
        if not (pullback and momentum):
            return None
        return dict(entry=cl, sl=min(recent_low, l) - 0.15 * av,
                    kind="Engulfing" if engulf else "Pin Bar", deep=deep)
    else:
        engulf = cl < o and pc > po and body > prng * 0.8 and cl < df["low"].iloc[i - 2]
        pin = body > 0 and uw >= body * 2.0 and lw <= body * 0.4
        if not (engulf or pin):
            return None
        pullback = recent_high >= ema20 - 0.6 * av and cl < ema50
        deep = recent_high >= ema50 - 0.3 * av
        momentum = cl < ema20
        if not (pullback and momentum):
            return None
        return dict(entry=cl, sl=max(recent_high, h) + 0.15 * av,
                    kind="Engulfing" if engulf else "Pin Bar", deep=deep)


def in_session_hour(hr: int) -> bool:
    return 7 <= hr < 21          # London 07-16 + NY 12-21 (UTC)


# ── outcome simulation (mirror of monitor_trades) ───────────────────────────
def simulate(df, i, direction, entry, sl, tp_mult):
    """Walk bars from i forward. Returns (result, r_multiple) or None if unresolved."""
    risk = abs(entry - sl)
    buy = direction == "BUY"
    tp1 = entry + risk if buy else entry - risk
    tp2 = entry + risk * tp_mult if buy else entry - risk * tp_mult
    stop, tp1_hit = sl, False
    for j in range(i, len(df)):
        hi, lo = df["high"].iloc[j], df["low"].iloc[j]
        if (lo <= stop) if buy else (hi >= stop):        # pessimistic: stop first
            if tp1_hit:
                return ("breakeven", 0.0)
            return ("loss", -1.0)
        if not tp1_hit and ((hi >= tp1) if buy else (lo <= tp1)):
            tp1_hit, stop = True, entry
        if (hi >= tp2) if buy else (lo <= tp2):
            return ("win", float(tp_mult))
    return None                                          # still open at data end


# ── candidate generation per symbol ─────────────────────────────────────────
def gen_candidates(name, entry_tf="30m"):
    ticker, pip = WATCHLIST[name]
    d1 = fetch(ticker, "1d", "2y")
    h1 = fetch(ticker, "60m", "60d")
    df = h1 if entry_tf == "60m" else fetch(ticker, "30m", "60d")
    if d1 is None or h1 is None or df is None or len(df) < WARMUP_BARS + 20:
        return [], None
    h4 = to_h4(h1)
    htf = HTFBias(d1, h4)
    e20, e50, a = ema(df["close"], 20), ema(df["close"], 50), atr(df, 14)

    out = []
    for i in range(WARMUP_BARS, len(df)):
        t = df.index[i]
        price_now = df["close"].iloc[i - 1]
        ob = htf.at(t, price_now)
        if ob == "none":
            continue
        direction = "BUY" if ob == "bull" else "SELL"
        sig = candle_signal_at(df, e20, e50, a, i, direction)
        if sig is None:
            continue
        if abs(sig["entry"] - sig["sl"]) <= 0:
            continue
        if not clean_structure(df.iloc[:i + 1], direction):
            continue
        in_sess = in_session_hour(t.hour)
        conf = score_setup({"kind": sig["kind"], "deep": sig["deep"]}, in_sess)
        if conf < CONF_THRESHOLD:
            continue
        out.append(dict(symbol=name, i=i, time=t, direction=direction,
                        entry=sig["entry"], sl=sig["sl"], kind=sig["kind"],
                        conf=conf, hour=t.hour))
    return out, df


# ── portfolio pass: cooldown + daily cap, then outcomes ─────────────────────
def run_variant(cands_by_sym, frames, session_gate=False, tp_mult=2.0,
                cooldown_h=COOLDOWN_HOURS, daily_cap=MAX_SIGNALS_PER_DAY):
    all_c = sorted([c for cl in cands_by_sym.values() for c in cl],
                   key=lambda c: c["time"])
    last_fire: dict = {}
    day_count: dict = {}
    taken = []
    for c in all_c:
        if session_gate and not in_session_hour(c["hour"]):
            continue
        lf = last_fire.get(c["symbol"])
        if lf is not None and (c["time"] - lf).total_seconds() < cooldown_h * 3600:
            continue
        day = str(c["time"].date())
        if day_count.get(day, 0) >= daily_cap:
            continue
        res = simulate(frames[c["symbol"]], c["i"], c["direction"],
                       c["entry"], c["sl"], tp_mult)
        last_fire[c["symbol"]] = c["time"]
        day_count[day] = day_count.get(day, 0) + 1
        if res is None:
            continue                                     # unresolved at data end
        taken.append({**c, "result": res[0], "r": res[1]})
    return taken


def report(label, trades):
    w = sum(1 for t in trades if t["result"] == "win")
    l = sum(1 for t in trades if t["result"] == "loss")
    b = sum(1 for t in trades if t["result"] == "breakeven")
    net = sum(t["r"] for t in trades)
    dec = w + l
    wr = w / dec * 100 if dec else 0
    ev = net / len(trades) if trades else 0
    print(f"{label:34} n={len(trades):3d}  {w}W {l}L {b}BE  "
          f"WR {wr:4.1f}%  net {net:+6.1f}R  EV {ev:+.3f}R/trade")
    return dict(n=len(trades), w=w, l=l, be=b, wr=wr, net=net, ev=ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", help="comma-separated subset")
    args = ap.parse_args()
    syms = ([s.strip().upper() for s in args.symbols.split(",")]
            if args.symbols else list(WATCHLIST.keys()))

    print(f"Generating candidates for {len(syms)} symbols "
          f"(30M + 1H frames, ~60d window)...")
    c30, f30, c60, f60 = {}, {}, {}, {}
    for n in syms:
        c30[n], f30[n] = gen_candidates(n, "30m")
        time.sleep(0.2)
        c60[n], f60[n] = gen_candidates(n, "60m")
        time.sleep(0.2)
        print(f"  {n}: {len(c30[n])} 30M candidates, {len(c60[n])} 1H candidates")

    f30 = {k: v for k, v in f30.items() if v is not None}
    f60 = {k: v for k, v in f60.items() if v is not None}

    print("\n=== VARIANTS (same window, same rules unless stated) ===")
    results = {}
    results["A"] = report("A: LIVE 30M, all pairs",
                          run_variant(c30, f30))
    results["B"] = report("B: 30M + session gate 07-21 UTC",
                          run_variant(c30, f30, session_gate=True))
    results["C"] = report("C: 1H entries (old strategy)",
                          run_variant(c60, f60))
    results["D"] = report("D: 1H + session gate",
                          run_variant(c60, f60, session_gate=True))
    results["E"] = report("E: 30M + session + TP2=3R",
                          run_variant(c30, f30, session_gate=True, tp_mult=3.0))

    with open("backtest_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nsaved -> backtest_results.json")


if __name__ == "__main__":
    main()
