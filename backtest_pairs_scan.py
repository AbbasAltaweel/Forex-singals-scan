"""
Pairs Scan Backtest -- one-off research tool, NOT part of any live bot.

Purpose: run the exact same signal logic used by AbbasBot / AbbasBot Pairs
against a wide set of forex CROSS pairs (not the 7 majors already tested in
backtest.py) to find out which of them show a genuine backtested edge,
before adding any of them to the live AbbasBot Pairs scan list.

SCOPE HONESTY (same caveats as backtest.py):
  - Structure-based stops, the 1.5R trailing tier, and correlation/volatility
    tagging are backtested properly here.
  - News/holiday filter and H4/Daily trend confirmation are NOT backtested
    (same reasons as backtest.py -- no historical calendar archive, and
    multi-timeframe alignment is a bigger undertaking).
  - Every trade is taken regardless of flags, matching live "warn, don't
    block" behavior.
  - This is ONE historical window. A positive result here is evidence, not
    a guarantee the edge holds going forward -- same as every other backtest
    run in this project.

Run manually from the Actions tab (workflow_dispatch). Reuses the SAME repo
secrets already configured (no new secrets needed):
  TWELVE_DATA_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
import time
import requests

# Cross/minor pairs NOT already covered by backtest.py's major-pair run.
# (Majors already tested previously: EUR/USD, GBP/USD, USD/JPY, USD/CHF,
#  AUD/USD, USD/CAD, NZD/USD, XAU/USD -- results already known for those.)
PAIRS = [
    ("EUR/GBP", "EUR/GBP"), ("EUR/JPY", "EUR/JPY"), ("GBP/JPY", "GBP/JPY"),
    ("EUR/CHF", "EUR/CHF"), ("EUR/AUD", "EUR/AUD"), ("EUR/CAD", "EUR/CAD"),
    ("EUR/NZD", "EUR/NZD"), ("GBP/CHF", "GBP/CHF"), ("GBP/AUD", "GBP/AUD"),
    ("GBP/CAD", "GBP/CAD"), ("GBP/NZD", "GBP/NZD"), ("AUD/JPY", "AUD/JPY"),
    ("CAD/JPY", "CAD/JPY"), ("CHF/JPY", "CHF/JPY"), ("NZD/JPY", "NZD/JPY"),
    ("AUD/CAD", "AUD/CAD"), ("AUD/CHF", "AUD/CHF"), ("AUD/NZD", "AUD/NZD"),
    ("NZD/CAD", "NZD/CAD"), ("NZD/CHF", "NZD/CHF"), ("CAD/CHF", "CAD/CHF"),
    ("XAG/USD", "SILVER (XAG/USD)"),
]

PAIR_CURRENCIES = {symbol: set(symbol.replace("XAG", "XAG").split("/")) for symbol, _ in PAIRS}

# Rough retail spread estimates in price units (used only to subtract
# realistic cost from each trade's R result -- same approach as backtest.py).
SPREADS = {
    "EUR/GBP": 0.00015, "EUR/JPY": 0.02, "GBP/JPY": 0.03,
    "EUR/CHF": 0.00020, "EUR/AUD": 0.00030, "EUR/CAD": 0.00030,
    "EUR/NZD": 0.00040, "GBP/CHF": 0.00035, "GBP/AUD": 0.00045,
    "GBP/CAD": 0.00040, "GBP/NZD": 0.00060, "AUD/JPY": 0.02,
    "CAD/JPY": 0.02, "CHF/JPY": 0.02, "NZD/JPY": 0.025,
    "AUD/CAD": 0.00025, "AUD/CHF": 0.00025, "AUD/NZD": 0.00035,
    "NZD/CAD": 0.00030, "NZD/CHF": 0.00030, "CAD/CHF": 0.00025,
    "XAG/USD": 0.03,
}

WINDOW = 100
RR_TP1, RR_TP2 = 1.0, 2.0
SCALP_PCT_THRESHOLD = 0.4
R_TP2, R_SL, R_BE, R_TRAIL_BE = 2.0, -1.0, 0.5, 0.75
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


def fetch_history(symbol, api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1h", "outputsize": 5000, "apikey": api_key}
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if data.get("status") == "error" or "code" in data:
        raise RuntimeError(data.get("message", "API error"))
    if "values" not in data:
        raise RuntimeError("No data returned")
    values = list(reversed(data["values"]))
    highs = [float(v["high"]) for v in values]
    lows = [float(v["low"]) for v in values]
    closes = [float(v["close"]) for v in values]
    return highs, lows, closes


def check_open_trade(trade, last_high, last_low, use_trailing=True):
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

    if use_trailing and not trade.get("trail_hit"):
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


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    if r.status_code != 200:
        print(f"Telegram send failed: {r.status_code} {r.text}", file=sys.stderr)


def run_simulation(data):
    min_len = min(len(d["closes"]) for d in data.values())
    open_trades = {}
    all_trades = []

    for i in range(60, min_len):
        for symbol in list(open_trades.keys()):
            d = data[symbol]
            result, still_open = check_open_trade(open_trades[symbol], d["highs"][i], d["lows"][i], use_trailing=True)
            if not still_open:
                trade = open_trades.pop(symbol)
                result_type, base_r = result
                spread = SPREADS.get(symbol, 0.0)
                spread_r = spread / trade["risk_dist"] if trade["risk_dist"] else 0
                all_trades.append({
                    "symbol": symbol, "pair": d["label"], "type": trade["trade_type"],
                    "result": result_type, "r": base_r - spread_r, "flags": trade["flags"],
                })

        for symbol, d in data.items():
            if symbol in open_trades:
                continue
            window_lo = max(0, i - WINDOW + 1)
            h_win, l_win, c_win = d["highs"][window_lo:i + 1], d["lows"][window_lo:i + 1], d["closes"][window_lo:i + 1]
            sig = build_signal(h_win, l_win, c_win)
            if not (sig["direction"] and sig["plan"] and sig["trade_type"] == "Swing"):
                continue

            flags = []
            if check_correlation(symbol, sig["direction"], open_trades):
                flags.append("correlated")
            if is_volatility_spike(h_win, l_win, c_win):
                flags.append("vol_spike")

            p = sig["plan"]
            open_trades[symbol] = {
                "direction": sig["direction"], "entry": p["entry"], "sl": p["sl"],
                "tp1": p["tp1"], "tp2": p["tp2"], "tp1_hit": False, "trail_hit": False,
                "trade_type": sig["trade_type"], "risk_dist": abs(p["entry"] - p["sl"]),
                "flags": flags,
            }

    return all_trades


def profit_factor(trades):
    gains = sum(t["r"] for t in trades if t["r"] > 0)
    losses = abs(sum(t["r"] for t in trades if t["r"] < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def main():
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    data = {}
    fetch_errors = []
    for symbol, label in PAIRS:
        try:
            highs, lows, closes = fetch_history(symbol, api_key)
            data[symbol] = {"highs": highs, "lows": lows, "closes": closes, "label": label}
            print(f"Fetched {symbol}: {len(closes)} candles")
        except Exception as e:
            fetch_errors.append(f"{label}: {e}")
            print(f"Failed {symbol}: {e}", file=sys.stderr)
        time.sleep(7)

    if not data:
        send_telegram(tg_token, tg_chat, "<b>🔬 Pairs Scan Backtest</b>\n\nAll pair fetches failed:\n" + "\n".join(fetch_errors))
        return

    all_trades = run_simulation(data)

    pair_stats = []
    for symbol, d in data.items():
        p_trades = [t for t in all_trades if t["symbol"] == symbol]
        wins = sum(1 for t in p_trades if t["result"] in ("tp2", "breakeven_trail", "breakeven_notrail"))
        wr = (wins / len(p_trades) * 100) if p_trades else 0
        total_r = sum(t["r"] for t in p_trades)
        pf = profit_factor(p_trades) if p_trades else 0.0
        pair_stats.append((symbol, d["label"], len(p_trades), wr, total_r, pf))

    # Only pairs with a meaningful sample get ranked; small samples are noise.
    ranked = sorted([p for p in pair_stats if p[2] >= 15], key=lambda p: p[4], reverse=True)
    unranked = sorted([p for p in pair_stats if p[2] < 15], key=lambda p: p[2], reverse=True)

    positive = [p for p in ranked if p[4] > 0]
    negative = [p for p in ranked if p[4] <= 0]

    lines = []
    if positive:
        lines.append("<b>✅ Positive edge (candidates to add):</b>")
        for symbol, label, count, wr, total_r, pf in positive:
            pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
            lines.append(f"{label}: {count} trades, {wr:.0f}% WR, +{total_r:.1f}R, PF {pf_str}")
        lines.append("")
    if negative:
        lines.append("<b>❌ Negative over this window (not recommended):</b>")
        for symbol, label, count, wr, total_r, pf in negative:
            pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
            lines.append(f"{label}: {count} trades, {wr:.0f}% WR, {total_r:.1f}R, PF {pf_str}")
        lines.append("")
    if unranked:
        lines.append("<b>⚠️ Too few trades to trust (sample &lt;15):</b>")
        for symbol, label, count, wr, total_r, pf in unranked:
            lines.append(f"{label}: {count} trades, {'+' if total_r>=0 else ''}{total_r:.1f}R")
        lines.append("")

    total_trades = len(all_trades)
    summary = (
        f"<b>🔬 Pairs Scan Backtest</b> (cross/minor pairs, not the majors already tested)\n\n"
        f"Total trades across {len(data)} pairs: {total_trades}\n\n"
        + "\n".join(lines) +
        f"\n<i>Scalp trades excluded. Spread cost subtracted. News/holiday filter and "
        f"H4/Daily confirmation not included (same limitation as the main backtest). "
        f"Past performance does not guarantee future results -- this is one historical "
        f"window, treat it as evidence to weigh, not a verdict.</i>"
    )
    if fetch_errors:
        summary += "\n\n<b>Fetch errors:</b>\n" + "\n".join(fetch_errors)

    send_telegram(tg_token, tg_chat, summary)
    print(summary)


if __name__ == "__main__":
    main()
