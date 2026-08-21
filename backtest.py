"""
Backtest: replays the EXACT signal logic from scanner.py against ~7-8 months
of historical hourly data, to show what the strategy would have actually
produced -- real win rate, total R, breakdown by pair and by scalp/swing.

This is a one-off analysis, not part of the live 15-minute scan. Run it
manually from the Actions tab whenever you want a fresh check.

Required environment variables (same secrets already in your repo):
  TWELVE_DATA_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
import time
import requests

PAIRS = [
    ("EUR/USD", "EUR/USD"),
    ("GBP/USD", "GBP/USD"),
    ("USD/JPY", "USD/JPY"),
    ("USD/CHF", "USD/CHF"),
    ("AUD/USD", "AUD/USD"),
    ("USD/CAD", "USD/CAD"),
    ("NZD/USD", "NZD/USD"),
    ("XAU/USD", "GOLD (XAU/USD)"),
]

WINDOW = 100  # same rolling window the live scanner uses per decision
ATR_STOP_MULTIPLIER = 1.5
RR_TP1, RR_TP2 = 1.0, 2.0
SCALP_PCT_THRESHOLD = 0.4
R_TP2, R_SL, R_BE = 2.0, -1.0, 0.5


# ---------- identical indicator math to scanner.py ----------
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
        # momentum confirmation (not reversal) -- keeps RSI aligned with
        # trend/MACD instead of contradicting them
        if rsi_val > 50:
            score += 1
        else:
            score -= 1
    if macd_val:
        score += 1 if macd_val["macd"] > macd_val["signal"] else -1

    direction = "buy" if score == 3 else ("sell" if score == -3 else None)

    plan, trade_type = None, None
    if direction and atr_val:
        stop_dist = atr_val * ATR_STOP_MULTIPLIER
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


def fetch_history(symbol, api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1h", "outputsize": 5000, "apikey": api_key}
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if data.get("status") == "error" or "code" in data:
        raise RuntimeError(data.get("message", "API error"))
    if "values" not in data:
        raise RuntimeError("No data returned")
    values = list(reversed(data["values"]))  # oldest -> newest
    highs = [float(v["high"]) for v in values]
    lows = [float(v["low"]) for v in values]
    closes = [float(v["close"]) for v in values]
    return highs, lows, closes


def simulate_pair(label, highs, lows, closes):
    """Walk forward bar by bar, replaying the exact live decision + tracking logic."""
    trades_closed = []
    open_trade = None

    n = len(closes)
    start = max(60, 0)  # warmup for SMA50/MACD/ATR

    for i in range(start, n):
        window_lo = max(0, i - WINDOW + 1)
        h_win = highs[window_lo:i + 1]
        l_win = lows[window_lo:i + 1]
        c_win = closes[window_lo:i + 1]

        if open_trade is None:
            sig = build_signal(h_win, l_win, c_win)
            if sig["direction"] and sig["plan"]:
                p = sig["plan"]
                open_trade = {
                    "direction": sig["direction"], "entry": p["entry"], "sl": p["sl"],
                    "tp1": p["tp1"], "tp2": p["tp2"], "tp1_hit": False,
                    "trade_type": sig["trade_type"], "opened_idx": i,
                }
            continue

        # check this candle against the open trade (same logic as live check_open_trade)
        last_high, last_low = highs[i], lows[i]
        d, entry, sl, tp1, tp2 = open_trade["direction"], open_trade["entry"], open_trade["sl"], open_trade["tp1"], open_trade["tp2"]

        if not open_trade["tp1_hit"]:
            hit_tp1 = (last_high >= tp1) if d == "buy" else (last_low <= tp1)
            hit_sl = (last_low <= sl) if d == "buy" else (last_high >= sl)
            if hit_sl:
                trades_closed.append({"pair": label, "type": open_trade["trade_type"], "result": "sl", "r": R_SL})
                open_trade = None
            elif hit_tp1:
                open_trade["tp1_hit"] = True
                open_trade["effective_sl"] = entry
        else:
            eff_sl = open_trade.get("effective_sl", entry)
            hit_tp2 = (last_high >= tp2) if d == "buy" else (last_low <= tp2)
            hit_be = (last_low <= eff_sl) if d == "buy" else (last_high >= eff_sl)
            if hit_tp2:
                trades_closed.append({"pair": label, "type": open_trade["trade_type"], "result": "tp2", "r": R_TP2})
                open_trade = None
            elif hit_be:
                trades_closed.append({"pair": label, "type": open_trade["trade_type"], "result": "breakeven", "r": R_BE})
                open_trade = None

    return trades_closed


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    if r.status_code != 200:
        print(f"Telegram send failed: {r.status_code} {r.text}", file=sys.stderr)


def main():
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    all_trades = []
    per_pair_lines = []

    for symbol, label in PAIRS:
        try:
            highs, lows, closes = fetch_history(symbol, api_key)
            trades = simulate_pair(label, highs, lows, closes)
            all_trades.extend(trades)
            wins = sum(1 for t in trades if t["result"] in ("tp2", "breakeven"))
            wr = (wins / len(trades) * 100) if trades else 0
            total_r = sum(t["r"] for t in trades)
            per_pair_lines.append(f"{label}: {len(trades)} trades, {wr:.0f}% win rate, {'+' if total_r>=0 else ''}{total_r:.1f}R")
        except Exception as e:
            per_pair_lines.append(f"{label}: error — {e}")
        time.sleep(7)

    if all_trades:
        wins = sum(1 for t in all_trades if t["result"] in ("tp2", "breakeven"))
        losses = sum(1 for t in all_trades if t["result"] == "sl")
        win_rate = wins / len(all_trades) * 100
        total_r = sum(t["r"] for t in all_trades)
        avg_r = total_r / len(all_trades)
        scalp_trades = [t for t in all_trades if t["type"] == "Scalp"]
        swing_trades = [t for t in all_trades if t["type"] == "Swing"]

        def stats_for(subset):
            if not subset:
                return "n/a"
            w = sum(1 for t in subset if t["result"] in ("tp2", "breakeven"))
            r = sum(t["r"] for t in subset)
            return f"{len(subset)} trades, {w/len(subset)*100:.0f}% win rate, {'+' if r>=0 else ''}{r:.1f}R"

        summary = (
            f"<b>🔬 Backtest Results</b>  (~7-8 months hourly data)\n\n"
            f"Total trades: {len(all_trades)}\n"
            f"Wins: {wins}  ·  Losses: {losses}\n"
            f"Win rate: {win_rate:.0f}%\n"
            f"Total R: {'+' if total_r>=0 else ''}{total_r:.2f}\n"
            f"Avg R/trade: {'+' if avg_r>=0 else ''}{avg_r:.2f}\n\n"
            f"Scalp: {stats_for(scalp_trades)}\n"
            f"Swing: {stats_for(swing_trades)}\n\n"
            f"<b>By pair:</b>\n" + "\n".join(per_pair_lines) + "\n\n"
            f"<i>Breakeven-after-TP1 counted as +0.5R. Past performance on historical "
            f"data does not guarantee future results.</i>"
        )
    else:
        summary = "<b>🔬 Backtest Results</b>\n\nNo trades were generated over the available history.\n\n" + "\n".join(per_pair_lines)

    send_telegram(tg_token, tg_chat, summary)
    print(summary)


if __name__ == "__main__":
    main()
