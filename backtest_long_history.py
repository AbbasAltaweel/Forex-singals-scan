"""
Long-History Backtest (OANDA) -- one-off research tool, NOT part of any live bot.

Fixes the biggest weakness of every earlier backtest: Twelve Data's free tier
only gave ~7 months of H1 data (and H1 vs H4 covered different calendar
windows). This pulls ~5 YEARS of candles from OANDA's free practice-account
API for BOTH H1 and H4, so the two timeframes are compared over the SAME
window, and results are broken down YEAR BY YEAR so a pair that only worked in
one stretch is visible.

Same strategy logic as the live bots (3/3 agreement, structure stops, 1:2 RR,
TP1 -> breakeven, 1.5R trailing tier). Spread cost subtracted. News filter and
H4/Daily confirmation tags are not simulated. One historical window is still
evidence, not a guarantee.

Required environment variables:
  OANDA_API_TOKEN       (practice-account API token)
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
Optional:
  OANDA_HOST            default https://api-fxpractice.oanda.com
  HISTORY_YEARS         default 5
"""

import os
import sys
import time
import datetime
import html
import re
import requests

PAIRS = [
    ("XAU_USD", "XAU/USD", "GOLD"),
    ("USD_CAD", "USD/CAD", "USD/CAD"),
    ("GBP_USD", "GBP/USD", "GBP/USD"),
    ("AUD_JPY", "AUD/JPY", "AUD/JPY"),
    ("EUR_USD", "EUR/USD", "EUR/USD"),
    ("USD_JPY", "USD/JPY", "USD/JPY"),
    ("AUD_USD", "AUD/USD", "AUD/USD"),
    ("USD_CHF", "USD/CHF", "USD/CHF"),
    ("NZD_USD", "NZD/USD", "NZD/USD"),
]

PAIR_CURRENCIES = {sym: ({"USD"} if sym == "XAU/USD" else set(sym.split("/"))) for _, sym, _ in PAIRS}

SPREADS = {
    "XAU/USD": 0.35, "USD/CAD": 0.00015, "GBP/USD": 0.00015, "AUD/JPY": 0.02,
    "EUR/USD": 0.00012, "USD/JPY": 0.015, "AUD/USD": 0.00015, "USD/CHF": 0.00018,
    "NZD/USD": 0.00020,
}

OANDA_HOST = os.environ.get("OANDA_HOST", "https://api-fxpractice.oanda.com")
HISTORY_YEARS = int(os.environ.get("HISTORY_YEARS", "5"))

WINDOW = 100
RR_TP1, RR_TP2 = 1.0, 2.0
SCALP_PCT_THRESHOLD = 0.4
# R ACCOUNTING (consistent "half closed at TP1" model, matching the bot's
# "bank partial at TP1, stop to breakeven" advice):
#   SL before TP1            = -1.0R
#   TP1 then stopped at BE   = 0.5*1R + 0.5*0   = +0.5R
#   TP1, trail, stopped 0.75 = 0.5*1R + 0.5*0.75 = +0.875R
#   TP1 then TP2             = 0.5*1R + 0.5*2R   = +1.5R
# Earlier backtests scored TP2 as +2.0R and trail as +0.75R while ALSO crediting
# +0.5R for breakeven exits -- mixing "full position" and "half position" math.
# That overstated every result by ~+0.08R per trade even on pure random data.
R_TP2, R_SL, R_BE, R_TRAIL_BE = 1.5, -1.0, 0.5, 0.875
VOLATILITY_SPIKE_MULTIPLIER = 2.0


def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema_series(values, period):
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = sum(values[:period]) / period
    out = [e]
    for v in values[period:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]
        gain = diff if diff > 0 else 0
        loss = -diff if diff < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(values):
    if len(values) < 35:
        return None
    ema12 = ema_series(values, 12)
    ema26 = ema_series(values, 26)
    if not ema12 or not ema26:
        return None
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[i + offset] - v for i, v in enumerate(ema26)]
    if len(macd_line) < 9:
        return None
    k = 2 / (9 + 1)
    signal = sum(macd_line[:9]) / 9
    signal_series = [signal]
    for v in macd_line[9:]:
        signal = v * k + signal * (1 - k)
        signal_series.append(signal)
    return {"macd": macd_line[-1], "signal": signal_series[-1]}


def atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def is_volatility_spike(highs, lows, closes):
    current_atr = atr(highs, lows, closes, 14)
    baseline_atr = atr(highs, lows, closes, 50)
    if not current_atr or not baseline_atr or baseline_atr == 0:
        return False
    return current_atr > baseline_atr * VOLATILITY_SPIKE_MULTIPLIER


def build_signal(highs, lows, closes):
    price = closes[-1]
    sma20, sma50 = sma(closes, 20), sma(closes, 50)
    rsi_val = rsi(closes, 14)
    macd_val = macd(closes)
    atr_val = atr(highs, lows, closes, 14)

    score = 0
    if sma20 and sma50:
        if price > sma20 > sma50:
            score += 1
        elif price < sma20 < sma50:
            score -= 1
    if rsi_val is not None:
        score += 1 if rsi_val > 50 else -1
    if macd_val:
        score += 1 if macd_val["macd"] > macd_val["signal"] else -1

    direction = "buy" if score == 3 else ("sell" if score == -3 else None)

    plan, trade_type = None, None
    if direction and atr_val:
        buffer = atr_val * 0.25
        swing_lookback = 20
        if direction == "buy":
            swing_low = min(lows[-swing_lookback:])
            structure_dist = price - (swing_low - buffer)
        else:
            swing_high = max(highs[-swing_lookback:])
            structure_dist = (swing_high + buffer) - price
        stop_dist = max(atr_val * 0.75, min(structure_dist, atr_val * 3.0))

        if direction == "buy":
            sl = price - stop_dist
            tp1, tp2 = price + stop_dist * RR_TP1, price + stop_dist * RR_TP2
        else:
            sl = price + stop_dist
            tp1, tp2 = price - stop_dist * RR_TP1, price - stop_dist * RR_TP2
        plan = {"entry": price, "sl": sl, "tp1": tp1, "tp2": tp2}
        tp2_pct = abs(stop_dist * RR_TP2) / price * 100
        trade_type = "Scalp" if tp2_pct < SCALP_PCT_THRESHOLD else "Swing"

    return {"direction": direction, "plan": plan, "trade_type": trade_type}


def check_correlation(symbol, direction, open_trades):
    my_currencies = PAIR_CURRENCIES.get(symbol, set())
    for open_symbol, t in open_trades.items():
        if open_symbol == symbol:
            continue
        their_currencies = PAIR_CURRENCIES.get(open_symbol, set())
        if (my_currencies & their_currencies) and t["direction"] == direction:
            return True
    return False


def check_open_trade(trade, last_high, last_low):
    direction = trade["direction"]
    entry, sl, tp1, tp2 = trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]
    risk_dist = trade["risk_dist"]

    if not trade["tp1_hit"]:
        hit_tp1 = (last_high >= tp1) if direction == "buy" else (last_low <= tp1)
        hit_sl = (last_low <= sl) if direction == "buy" else (last_high >= sl)
        if hit_sl:
            return ("sl", R_SL), False
        if hit_tp1:
            trade["tp1_hit"] = True
            trade["effective_sl"] = entry
        return None, True

    if not trade.get("trail_hit"):
        trail_level = entry + risk_dist * 1.5 if direction == "buy" else entry - risk_dist * 1.5
        hit_trail = (last_high >= trail_level) if direction == "buy" else (last_low <= trail_level)
        if hit_trail:
            trade["trail_hit"] = True
            trade["effective_sl"] = entry + risk_dist * 0.75 if direction == "buy" else entry - risk_dist * 0.75

    eff_sl = trade["effective_sl"]
    hit_tp2 = (last_high >= tp2) if direction == "buy" else (last_low <= tp2)
    hit_be = (last_low <= eff_sl) if direction == "buy" else (last_high >= eff_sl)

    if hit_tp2:
        return ("tp2", R_TP2), False
    if hit_be:
        if trade.get("trail_hit"):
            return ("breakeven_trail", R_TRAIL_BE), False
        return ("breakeven_notrail", R_BE), False
    return None, True


def profit_factor(trades):
    gains = sum(t["r"] for t in trades if t["r"] > 0)
    losses = abs(sum(t["r"] for t in trades if t["r"] < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def parse_time(value):
    """OANDA times come back as RFC3339 text ('2021-11-23T14:00:00.000000000Z')
    or UNIX seconds depending on settings. Accept both -> UNIX seconds (float)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    txt = str(value).rstrip("Z")
    if "." in txt:
        base, frac = txt.split(".", 1)
        txt = base + "." + frac[:6]
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in txt else "%Y-%m-%dT%H:%M:%S"
    return datetime.datetime.strptime(txt, fmt).replace(tzinfo=datetime.timezone.utc).timestamp()


def to_rfc3339(ts):
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def fetch_history(instrument, token, granularity):
    """Paginated OANDA candle download, oldest -> newest, complete candles only."""
    headers = {"Authorization": f"Bearer {token}"}
    start = time.time() - HISTORY_YEARS * 365.25 * 24 * 3600
    now = time.time()
    times, highs, lows, closes = [], [], [], []
    cursor = start
    while cursor < now:
        params = {"granularity": granularity, "price": "M", "count": 5000,
                  "from": f"{cursor:.0f}"}
        r = None
        for attempt in range(4):
            r = requests.get(f"{OANDA_HOST}/v3/instruments/{instrument}/candles",
                             headers=headers, params=params, timeout=60)
            if r.status_code == 200 or r.status_code < 500 and r.status_code != 429:
                break
            time.sleep(3 * (attempt + 1))   # 5xx / rate limit: wait and retry
        if r.status_code != 200:
            body = re.sub(r"<[^>]+>", " ", r.text)
            body = " ".join(body.split())[:100]
            raise RuntimeError(f"HTTP {r.status_code} {body}")
        candles = r.json().get("candles", [])
        new = [c for c in candles if c.get("complete") and parse_time(c["time"]) > (times[-1] if times else 0)]
        if not candles:
            break
        for c in new:
            times.append(parse_time(c["time"]))
            highs.append(float(c["mid"]["h"]))
            lows.append(float(c["mid"]["l"]))
            closes.append(float(c["mid"]["c"]))
        last_time = parse_time(candles[-1]["time"])
        if last_time <= cursor:
            break
        cursor = last_time + 1
        if len(candles) < 5000:
            break
        time.sleep(0.3)
    if len(closes) < 200:
        raise RuntimeError("Too little data returned")
    return times, highs, lows, closes


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


def run_simulation(symbol, d):
    """Single-pair walk-forward replay of the live logic. Returns closed trades
    with the calendar year each closed in."""
    trades, open_trade = [], None
    n = len(d["closes"])
    for i in range(60, n):
        if open_trade:
            result, still_open = check_open_trade(open_trade, d["highs"][i], d["lows"][i])
            if not still_open:
                result_type, base_r = result
                spread_r = SPREADS.get(symbol, 0.0) / open_trade["risk_dist"] if open_trade["risk_dist"] else 0
                year = datetime.datetime.utcfromtimestamp(d["times"][i]).year
                trades.append({"result": result_type, "r": base_r - spread_r, "year": year})
                open_trade = None
        if open_trade is None:
            lo = max(0, i - WINDOW + 1)
            h, l, c = d["highs"][lo:i + 1], d["lows"][lo:i + 1], d["closes"][lo:i + 1]
            sig = build_signal(h, l, c)
            if sig["direction"] and sig["plan"] and sig["trade_type"] == "Swing":
                p = sig["plan"]
                open_trade = {"direction": sig["direction"], "entry": p["entry"], "sl": p["sl"],
                              "tp1": p["tp1"], "tp2": p["tp2"], "tp1_hit": False, "trail_hit": False,
                              "risk_dist": abs(p["entry"] - p["sl"])}
    return trades


def summarize(trades):
    if not trades:
        return "no trades", 0, 0.0
    wins = sum(1 for t in trades if t["result"] in ("tp2", "breakeven_trail", "breakeven_notrail"))
    total_r = sum(t["r"] for t in trades)
    pf = profit_factor(trades)
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return f"{len(trades)} trades, {wins/len(trades)*100:.0f}% WR, {total_r:+.1f}R, PF {pf_s}", len(trades), total_r


def main():
    token = "".join(os.environ["OANDA_API_TOKEN"].split())  # strip any stray spaces/newlines from the secret
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"].strip()
    tg_chat = os.environ["TELEGRAM_CHAT_ID"].strip()

    blocks, errors = [], []
    for instrument, symbol, label in PAIRS:
        lines = [f"<b>{label}</b>"]
        for gran in ("H1", "H4"):
            try:
                times, highs, lows, closes = fetch_history(instrument, token, gran)
                d = {"times": times, "highs": highs, "lows": lows, "closes": closes}
                trades = run_simulation(symbol, d)
                text, n, total = summarize(trades)
                by_year = {}
                for t in trades:
                    by_year[t["year"]] = by_year.get(t["year"], 0.0) + t["r"]
                years = " ".join(f"{y}:{r:+.1f}" for y, r in sorted(by_year.items()))
                span = (times[-1] - times[0]) / (365.25 * 24 * 3600)
                lines.append(f"  {gran} ({span:.1f}y): {text}")
                if years:
                    lines.append(f"     by year R: {years}")
                print(f"{label} {gran}: {text}")
            except Exception as e:
                errors.append(f"{label} {gran}: {html.escape(str(e)[:200])}")
                lines.append(f"  {gran}: failed ({html.escape(str(e)[:120])})")
        blocks.append("\n".join(lines))

    header_msg = (f"<b>Long-History Backtest ({HISTORY_YEARS}y, OANDA)</b>\n"
                  "Same window for H1 and H4. Positive in most years = real; "
                  "one big year carrying the total = not trusted.\n")
    msg, parts = header_msg, []
    for b in blocks:
        if len(msg) + len(b) > 3800:
            parts.append(msg)
            msg = ""
        msg += "\n" + b + "\n"
    parts.append(msg)
    if errors:
        parts.append("<b>Errors:</b>\n" + "\n".join(errors))
    parts[-1] += ("\n<i>Spread subtracted; news/H4-daily tags not simulated. "
                  "Evidence, not a guarantee.</i>")
    for p in parts:
        send_telegram(tg_token, tg_chat, p)
        print(p)


if __name__ == "__main__":
    main()
