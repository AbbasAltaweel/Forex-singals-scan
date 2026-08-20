"""
Forex + Gold signal scanner with trade tracking -> Telegram alerts.

This version REMEMBERS open trades between runs (state saved to
trades_state.json, committed back to the repo by the workflow) and:

  - Opens a new trade only on full 3/3 indicator agreement (trend + RSI +
    MACD all aligned) -- higher conviction, fewer but stronger signals.
  - Tags each new trade as Scalp or Swing based on how large the target
    move is relative to price.
  - Every run, re-checks each OPEN trade's most recent candle against its
    levels:
      * TP1 hit  -> tells you to bank partial profit and move stop to
        breakeven, keeps the trade open (now risk-free) targeting TP2.
      * TP2 hit  -> full target reached, closes the trade.
      * SL hit (before TP1)   -> stopped out for a loss, closes the trade.
      * Breakeven hit (after TP1) -> closed flat, no loss, no further gain.
  - Only opens ONE trade per pair at a time (won't stack conflicting
    signals on the same pair).

Required environment variables (GitHub Actions secrets):
  TWELVE_DATA_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
import time
import json
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

STATE_FILE = "trades_state.json"
REQUEST_GAP_SECONDS = 7
ATR_STOP_MULTIPLIER = 1.5
RR_TP1 = 1.0
RR_TP2 = 2.0
SCALP_PCT_THRESHOLD = 0.4  # TP2 distance as % of price; below = scalp, above = swing


# ---------- indicators (oldest -> newest order) ----------
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
    notes = []

    if sma20 and sma50:
        if price > sma20 > sma50:
            score += 1
            notes.append("Trend: bullish")
        elif price < sma20 < sma50:
            score -= 1
            notes.append("Trend: bearish")
        else:
            notes.append("Trend: mixed")
    else:
        notes.append("Trend: insufficient data")

    if rsi_val is not None:
        if rsi_val < 30:
            score += 1
            notes.append(f"RSI {rsi_val:.1f}: oversold")
        elif rsi_val > 70:
            score -= 1
            notes.append(f"RSI {rsi_val:.1f}: overbought")
        else:
            notes.append(f"RSI {rsi_val:.1f}: neutral")
    else:
        notes.append("RSI: insufficient data")

    if macd_val:
        if macd_val["macd"] > macd_val["signal"]:
            score += 1
            notes.append("MACD: bullish crossover")
        else:
            score -= 1
            notes.append("MACD: bearish crossover")
    else:
        notes.append("MACD: insufficient data")

    # Full-agreement filter: require all 3 indicators aligned (score == 3 or -3)
    direction = None
    if score == 3:
        direction = "buy"
    elif score == -3:
        direction = "sell"

    plan = None
    trade_type = None
    if direction and atr_val:
        stop_dist = atr_val * ATR_STOP_MULTIPLIER
        if direction == "buy":
            sl = price - stop_dist
            tp1 = price + stop_dist * RR_TP1
            tp2 = price + stop_dist * RR_TP2
        else:
            sl = price + stop_dist
            tp1 = price - stop_dist * RR_TP1
            tp2 = price - stop_dist * RR_TP2
        plan = {"entry": price, "sl": sl, "tp1": tp1, "tp2": tp2}
        tp2_pct = abs(stop_dist * RR_TP2) / price * 100
        trade_type = "Scalp" if tp2_pct < SCALP_PCT_THRESHOLD else "Swing"

    return {
        "price": price, "score": score, "notes": notes,
        "direction": direction, "plan": plan, "trade_type": trade_type,
    }


def fetch_ohlc(symbol, api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1h", "outputsize": 100, "apikey": api_key}
    r = requests.get(url, params=params, timeout=20)
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


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    if r.status_code != 200:
        print(f"Telegram send failed: {r.status_code} {r.text}", file=sys.stderr)


def decimals_for(symbol):
    if symbol == "USD/JPY":
        return 3
    if symbol == "XAU/USD":
        return 2
    return 5


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {"trades": {}}
    return {"trades": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_open_trade(trade, highs, lows, closes, d):
    """Check the latest candle against this trade's levels. Returns (status_msg or None, updated_trade, still_open)."""
    last_high, last_low = highs[-1], lows[-1]
    direction = trade["direction"]
    entry, sl, tp1, tp2 = trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]
    pair_label = trade["label"]

    if not trade.get("tp1_hit"):
        if direction == "buy":
            hit_tp1 = last_high >= tp1
            hit_sl = last_low <= sl
        else:
            hit_tp1 = last_low <= tp1
            hit_sl = last_high >= sl

        if hit_sl:
            msg = (f"🔴 <b>{pair_label}</b> — SL hit\n"
                   f"Trade closed at <code>{sl:.{d}f}</code> — stop-loss triggered.\n"
                   f"<i>Result: loss (-1R)</i>")
            return msg, trade, False
        if hit_tp1:
            trade["tp1_hit"] = True
            trade["effective_sl"] = entry  # move to breakeven
            msg = (f"🟡 <b>{pair_label}</b> — TP1 hit!\n"
                   f"Price reached <code>{tp1:.{d}f}</code>.\n"
                   f"👉 Suggest: bank partial profit here, move stop to breakeven (<code>{entry:.{d}f}</code>).\n"
                   f"Holding remainder for TP2 at <code>{tp2:.{d}f}</code>.")
            return msg, trade, True
        return None, trade, True
    else:
        effective_sl = trade.get("effective_sl", entry)
        if direction == "buy":
            hit_tp2 = last_high >= tp2
            hit_be = last_low <= effective_sl
        else:
            hit_tp2 = last_low <= tp2
            hit_be = last_high >= effective_sl

        if hit_tp2:
            msg = (f"🟢🟢 <b>{pair_label}</b> — TP2 hit! Full target reached\n"
                   f"Closed at <code>{tp2:.{d}f}</code>.\n"
                   f"<i>Result: win (+2R)</i>")
            return msg, trade, False
        if hit_be:
            msg = (f"⚪ <b>{pair_label}</b> — Stopped at breakeven\n"
                   f"Price came back to entry <code>{entry:.{d}f}</code> after TP1.\n"
                   f"<i>Result: flat (0R) — TP1 partial gain already banked</i>")
            return msg, trade, False
        return None, trade, True


def main():
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    state = load_state()
    trades = state.get("trades", {})

    update_blocks = []
    new_signal_blocks = []
    quiet_lines = []

    for symbol, label in PAIRS:
        try:
            highs, lows, closes = fetch_ohlc(symbol, api_key)
            d = decimals_for(symbol)

            if symbol in trades:
                msg, updated_trade, still_open = check_open_trade(trades[symbol], highs, lows, closes, d)
                if msg:
                    update_blocks.append(msg)
                if still_open:
                    trades[symbol] = updated_trade
                else:
                    del trades[symbol]
                    quiet_lines.append(f"{label}: trade closed, now flat")
            else:
                sig = build_signal(highs, lows, closes)
                if sig["direction"] and sig["plan"]:
                    p = sig["plan"]
                    trade = {
                        "label": label, "direction": sig["direction"],
                        "entry": p["entry"], "sl": p["sl"], "tp1": p["tp1"], "tp2": p["tp2"],
                        "tp1_hit": False, "trade_type": sig["trade_type"],
                        "opened_at": int(time.time()),
                    }
                    trades[symbol] = trade
                    dir_word = "BUY" if sig["direction"] == "buy" else "SELL"
                    emoji = "🟢🟢" if sig["direction"] == "buy" else "🔴🔴"
                    block = (
                        f"{emoji} <b>{label}</b> — <b>{dir_word}</b>  [{sig['trade_type']}]\n"
                        f"Entry: <code>{p['entry']:.{d}f}</code>\n"
                        f"SL: <code>{p['sl']:.{d}f}</code>\n"
                        f"TP1: <code>{p['tp1']:.{d}f}</code>  (1R)\n"
                        f"TP2: <code>{p['tp2']:.{d}f}</code>  (2R)\n"
                        f"Risk:Reward  1 : 2\n"
                        f"<i>Full agreement: {' · '.join(sig['notes'])}</i>"
                    )
                    new_signal_blocks.append(block)
                else:
                    quiet_lines.append(f"{label}: no full-agreement setup")
        except Exception as e:
            quiet_lines.append(f"⚠️ {label}: error — {e}")
        time.sleep(REQUEST_GAP_SECONDS)

    state["trades"] = trades
    save_state(state)

    parts = []
    if update_blocks:
        parts.append("<b>📌 Trade Updates</b>\n\n" + "\n\n".join(update_blocks))
    if new_signal_blocks:
        parts.append("<b>🎯 New High-Conviction Setups</b>\n\n" + "\n\n".join(new_signal_blocks))

    if parts:
        message = "\n\n---\n\n".join(parts)
        if quiet_lines:
            message += "\n\n<i>" + "; ".join(quiet_lines) + "</i>"
    else:
        open_count = len(trades)
        message = (f"<b>📊 Forex &amp; Gold Scan</b>\n\nNo new setups or level hits this cycle.\n"
                    f"Open trades being tracked: {open_count}\n\n" + "\n".join(quiet_lines))

    send_telegram(tg_token, tg_chat, message)
    print(message)


if __name__ == "__main__":
    main()
