"""
Trend-Following Research Backtest -- one-off research tool, NOT part of any live bot.

WHY THIS EXISTS
The 5-year test of the live bots' strategy (H1/H4, RSI + MACD + moving-average
agreement) showed no reliable edge. Published research on FX points somewhere
different: slow, low-turnover trend following (lookbacks of months, not hours),
sized by volatility. This script tests a SHORT, FIXED list of those classic rules
on up to ~19 years of DAILY data for 7 majors + gold, with a hard split:

    TRAIN (in-sample):   data before 2016-01-01   -> used to choose
    VALIDATE (out-of-sample): 2016-01-01 onward   -> the verdict, looked at once

The candidate list below was fixed BEFORE seeing any results. Nothing is tuned.

STRATEGIES (position is -1..+1 of a volatility-targeted unit, decided at the close
and earned over the NEXT day -- no look-ahead):
  S1  TSM-126   sign of the last 126-day return (~6 months)
  S2  TSM-252   sign of the last 252-day return (~12 months)
  S3  MA 50/200 long if 50-day average above 200-day average, else short
  S4  Donchian  55-day breakout in, 20-day opposite breakout out (turtle-style)
  S5  Ensemble  average of S1..S4 (smaller position when rules disagree)

SIZING / COSTS
  Each instrument targets 10% annual volatility (20-day realised vol, max 4x).
  Portfolio = equal-weight average across instruments.
  Cost = full quoted spread on every unit of position change (conservative).
  NOT MODELLED: swap / financing (interest-rate carry). Holding FX positions
  earns or pays the rate difference every day; published studies include it and
  it can move results by several % a year. Treat results as "before carry".

HOW TO READ THE RESULT
  Choose by TRAIN Sharpe only. The VALIDATE Sharpe is the verdict:
  above ~0.3 with most validate years positive = worth forward-testing on a demo;
  otherwise nothing was found. Published FX trend-following studies report out-of-
  sample Sharpe of roughly 0.1-0.5, i.e. a thin edge, ~40-45% winning trades and
  long flat or losing stretches -- NOT a high win-rate system.

Required environment variables:
  TWELVE_DATA_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
import re
import time
import math
import html
import datetime
import requests

INSTRUMENTS = [
    ("EUR/USD", "EUR/USD"), ("GBP/USD", "GBP/USD"), ("USD/JPY", "USD/JPY"),
    ("USD/CHF", "USD/CHF"), ("AUD/USD", "AUD/USD"), ("USD/CAD", "USD/CAD"),
    ("NZD/USD", "NZD/USD"), ("XAU/USD", "GOLD"),
]

# Quoted spread in price units (retail-style, deliberately a bit wide).
SPREADS = {
    "EUR/USD": 0.00012, "GBP/USD": 0.00015, "USD/JPY": 0.015, "USD/CHF": 0.00018,
    "AUD/USD": 0.00015, "USD/CAD": 0.00015, "NZD/USD": 0.00020, "XAU/USD": 0.35,
}

SPLIT_DATE = "2016-01-01"
TARGET_VOL = 0.10
VOL_LOOKBACK = 20
MAX_LEVERAGE = 4.0
TRADING_DAYS = 252
TD_URL = "https://api.twelvedata.com/time_series"
TD_GAP_SECONDS = 8


# ----------------------------------------------------------------- data ---
def td_request(params):
    last = None
    for attempt in range(5):
        r = requests.get(TD_URL, params=params, timeout=60)
        try:
            data = r.json()
        except Exception:
            data = {"status": "error", "code": r.status_code, "message": r.text[:100]}
        last = data
        if (data.get("status") == "error" and data.get("code") == 429) or r.status_code >= 500:
            time.sleep(30 * (attempt + 1))
            continue
        return data
    return last


def fetch_daily(symbol, api_key):
    """Newest 5000 daily candles (~19 years) returned oldest -> newest."""
    data = td_request({"symbol": symbol, "interval": "1day", "outputsize": 5000,
                       "order": "DESC", "apikey": api_key})
    if data.get("status") == "error":
        raise RuntimeError(f"API {data.get('code')} {' '.join(str(data.get('message', '')).split())[:100]}")
    values = data.get("values") or []
    if len(values) < 400:
        raise RuntimeError("Too little data returned")
    values = list(reversed(values))
    dates = [v["datetime"][:10] for v in values]
    highs = [float(v["high"]) for v in values]
    lows = [float(v["low"]) for v in values]
    closes = [float(v["close"]) for v in values]
    return dates, highs, lows, closes


# ------------------------------------------------------------- signals ----
# Every signal function returns one value per bar, using ONLY data up to and
# including that bar (verified by a truncation test in the test suite).
def sma_at(values, i, n):
    if i + 1 < n:
        return None
    return sum(values[i + 1 - n:i + 1]) / n


def sig_tsm(closes, lookback):
    out = []
    for i in range(len(closes)):
        if i < lookback:
            out.append(0.0)
        else:
            d = closes[i] - closes[i - lookback]
            out.append(1.0 if d > 0 else (-1.0 if d < 0 else 0.0))
    return out


def sig_ma(closes, fast=50, slow=200):
    out = []
    for i in range(len(closes)):
        f, s = sma_at(closes, i, fast), sma_at(closes, i, slow)
        if f is None or s is None:
            out.append(0.0)
        else:
            out.append(1.0 if f > s else (-1.0 if f < s else 0.0))
    return out


def sig_donchian(highs, lows, closes, entry=55, exit_=20):
    out, state = [], 0.0
    for i in range(len(closes)):
        if i < entry:
            out.append(0.0)
            continue
        hi_e = max(highs[i - entry:i])
        lo_e = min(lows[i - entry:i])
        hi_x = max(highs[i - exit_:i])
        lo_x = min(lows[i - exit_:i])
        c = closes[i]
        if state == 0.0:
            if c > hi_e:
                state = 1.0
            elif c < lo_e:
                state = -1.0
        elif state > 0:
            if c < lo_x:
                state = 0.0
                if c < lo_e:
                    state = -1.0
        else:
            if c > hi_x:
                state = 0.0
                if c > hi_e:
                    state = 1.0
        out.append(state)
    return out


STRATEGIES = ["S1 TSM-126", "S2 TSM-252", "S3 MA 50/200", "S4 Donchian", "S5 Ensemble"]


def raw_signals(highs, lows, closes):
    s1 = sig_tsm(closes, 126)
    s2 = sig_tsm(closes, 252)
    s3 = sig_ma(closes, 50, 200)
    s4 = sig_donchian(highs, lows, closes, 55, 20)
    s5 = [(a + b + c + d) / 4.0 for a, b, c, d in zip(s1, s2, s3, s4)]
    return [s1, s2, s3, s4, s5]


# ------------------------------------------------------------- engine -----
def daily_returns(closes):
    return [0.0] + [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]


def vol_scale(rets):
    """Leverage multiplier at each bar from the trailing realised vol (data up to that bar)."""
    out = []
    for i in range(len(rets)):
        if i < VOL_LOOKBACK:
            out.append(0.0)
            continue
        w = rets[i - VOL_LOOKBACK + 1:i + 1]
        m = sum(w) / len(w)
        sd = math.sqrt(sum((x - m) ** 2 for x in w) / (len(w) - 1))
        ann = sd * math.sqrt(TRADING_DAYS)
        out.append(min(MAX_LEVERAGE, TARGET_VOL / ann) if ann > 0 else 0.0)
    return out


def pnl_series(symbol, dates, closes, raw):
    """Daily P&L (fraction of capital). Position chosen at close i-1 earns return i."""
    rets = daily_returns(closes)
    scale = vol_scale(rets)
    pos = [r * s for r, s in zip(raw, scale)]
    spread = SPREADS.get(symbol, 0.0)
    out = {}
    for i in range(2, len(closes)):
        gross = pos[i - 1] * rets[i]
        cost = abs(pos[i - 1] - pos[i - 2]) * spread / closes[i - 1]
        out[dates[i]] = gross - cost
    return out


def portfolio(per_instrument):
    """Equal-weight across instruments that have a value on that date."""
    acc = {}
    for series in per_instrument.values():
        for d, v in series.items():
            acc.setdefault(d, []).append(v)
    return {d: sum(v) / len(v) for d, v in sorted(acc.items())}


def stats(series, start=None, end=None):
    items = [(d, v) for d, v in sorted(series.items())
             if (start is None or d >= start) and (end is None or d < end)]
    # drop the warm-up stretch where no position existed yet
    while items and items[0][1] == 0.0:
        items.pop(0)
    n = len(items)
    if n < 60:
        return None
    vals = [v for _, v in items]
    mean = sum(vals) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n - 1))
    sharpe = (mean / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    cum, peak, mdd = 0.0, 0.0, 0.0
    for v in vals:
        cum += v
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    years = {}
    months = {}
    for d, v in items:
        years[d[:4]] = years.get(d[:4], 0.0) + v
        months[d[:7]] = months.get(d[:7], 0.0) + v
    pos_months = sum(1 for v in months.values() if v > 0) / len(months) * 100
    return {"sharpe": sharpe, "ann_ret": mean * TRADING_DAYS, "mdd": mdd, "years": years,
            "pos_years": sum(1 for v in years.values() if v > 0), "n_years": len(years),
            "pos_months": pos_months, "n": n}


# ------------------------------------------------------------ reporting ---
def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = text[:4000]
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    if r.status_code != 200:
        print(f"Telegram HTML send failed: {r.status_code} {r.text}", file=sys.stderr)
        plain = html.unescape(re.sub(r"<[^>]+>", "", text))
        r2 = requests.post(url, data={"chat_id": chat_id, "text": plain}, timeout=20)
        if r2.status_code != 200:
            print(f"Telegram plain send failed: {r2.status_code} {r2.text}", file=sys.stderr)


def fmt_sh(s):
    return "  n/a" if s is None else f"{s['sharpe']:+.2f}"


def build_reports(per_strategy_instr, span_lines, errors):
    """per_strategy_instr[strategy_index][symbol_label] -> pnl dict"""
    msgs = []
    port = [portfolio(d) for d in per_strategy_instr]
    tr = [stats(p, None, SPLIT_DATE) for p in port]
    va = [stats(p, SPLIT_DATE, None) for p in port]

    head = ["<b>Trend-Following Research (daily, up to ~19y)</b>",
            f"Train = before {SPLIT_DATE}; Validate = {SPLIT_DATE} onward (the verdict).",
            "Portfolio = equal-weight of 8 instruments, 10% vol target each, spreads paid, "
            "<b>no swap/carry modelled</b>.", ""]
    table_head = "Strategy        TrainSh ValidSh Ret/yr  MaxDD  +Yrs  +Mths"
    rows = []
    for i, name in enumerate(STRATEGIES):
        t, v = tr[i], va[i]
        if v is None:
            rows.append(f"{name:<14} {fmt_sh(t):>7}   n/a")
            continue
        rows.append(f"{name:<14} {fmt_sh(t):>7} {fmt_sh(v):>7} {v['ann_ret']*100:+6.1f}% {v['mdd']*100:5.1f}% "
                    f"{v['pos_years']:>2}/{v['n_years']:<2} {v['pos_months']:4.0f}%")
    msgs.append("\n".join(head) + "\n<pre>" + table_head + "\n" + "\n".join(rows) + "</pre>")

    # per-instrument validate Sharpe grid
    labels = list(per_strategy_instr[0].keys())
    grid_head = "Validate Sharpe by instrument"
    cols = "Instrument  " + " ".join(f"S{i+1:<5}" for i in range(len(STRATEGIES)))
    grid = [cols]
    for lab in labels:
        cells = []
        for i in range(len(STRATEGIES)):
            s = stats(per_strategy_instr[i][lab], SPLIT_DATE, None)
            cells.append(f"{fmt_sh(s):>6}")
        grid.append(f"{lab:<11} " + " ".join(cells))
    msgs.append(f"<b>{grid_head}</b>\n<pre>" + "\n".join(grid) + "</pre>")

    # year-by-year for the ensemble and the best-train strategy
    best_i = max(range(len(STRATEGIES)), key=lambda i: tr[i]["sharpe"] if tr[i] else -9)
    pick = sorted({4, best_i})
    ylines = []
    for i in pick:
        v_all = stats(port[i])
        if not v_all:
            continue
        yrs = " ".join(f"{y[2:]}:{r*100:+.0f}" for y, r in sorted(v_all["years"].items()))
        ylines.append(f"<b>{STRATEGIES[i]}</b> yearly % (10% vol/instr):\n{yrs}")
    verdict_i = best_i
    v = va[verdict_i]
    if v is None:
        verdict = "Not enough validate data."
    elif v["sharpe"] >= 0.3 and v["pos_years"] / v["n_years"] >= 0.6:
        verdict = (f"Best on TRAIN was <b>{STRATEGIES[verdict_i]}</b>; it held up on validate "
                   f"(Sharpe {v['sharpe']:+.2f}, {v['pos_years']}/{v['n_years']} positive years). "
                   "Worth forward-testing on a demo -- not proof.")
    else:
        verdict = (f"Best on TRAIN was <b>{STRATEGIES[verdict_i]}</b>, but on validate it scored "
                   f"Sharpe {v['sharpe']:+.2f} with {v['pos_years']}/{v['n_years']} positive years: "
                   "no strategy cleared the bar. Nothing found here is ready for real money.")
    msgs.append("\n\n".join(ylines) + f"\n\n<b>Verdict:</b> {verdict}\n\n"
                "<i>Reference: a published walk-forward study of FX trend-following found a median "
                "out-of-sample Sharpe of ~0.14 across pairs and rules. Carry/swap excluded here.</i>")

    msgs.append("<b>Data used</b>\n<pre>" + "\n".join(span_lines) + "</pre>"
                + ("\n<b>Errors:</b>\n" + "\n".join(errors) if errors else ""))
    return msgs


def main():
    api_key = "".join(os.environ["TWELVE_DATA_API_KEY"].split())
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    tg_chat = os.environ["TELEGRAM_CHAT_ID"].strip()

    per_strategy_instr = [dict() for _ in STRATEGIES]
    span_lines, errors = [], []
    for symbol, label in INSTRUMENTS:
        try:
            dates, highs, lows, closes = fetch_daily(symbol, api_key)
        except Exception as e:
            errors.append(f"{label}: {html.escape(str(e)[:150])}")
            time.sleep(TD_GAP_SECONDS)
            continue
        time.sleep(TD_GAP_SECONDS)
        span_lines.append(f"{label:<8} {dates[0]} -> {dates[-1]} ({len(closes)} bars)")
        for i, raw in enumerate(raw_signals(highs, lows, closes)):
            per_strategy_instr[i][label] = pnl_series(symbol, dates, closes, raw)

    if not per_strategy_instr[0]:
        send_telegram(tg_token, tg_chat, "<b>Trend research failed</b>\n" + "\n".join(errors))
        return
    for m in build_reports(per_strategy_instr, span_lines, errors):
        send_telegram(tg_token, tg_chat, m)
        print(m)


if __name__ == "__main__":
    main()
