"""
Forex + Gold signal scanner -> Telegram alerts with full trade plans.

Each run:
  1. Pulls the latest hourly OHLC candles for each pair from Twelve Data.
  2. Computes SMA20/50, RSI(14), MACD(12,26,9), and ATR(14) on the candles.
  3. Combines SMA/RSI/MACD into a composite signal per pair.
  4. For STRONG BUY / STRONG SELL only, builds a trade plan:
     Entry, Stop-Loss (1.5x ATR), TP1 (1R), TP2 (2R) -- a 1:2 risk/reward.
  5. Sends only the high-conviction setups to Telegram (skips weak/neutral
     pairs from the alert body, but lists them briefly at the bottom so you
     know the scan ran).

Required environment variables (GitHub Actions secrets):
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

REQUEST_GAP_SECONDS = 7
ATR_STOP_MULTIPLIER = 1.5  # stop distance = 1.5x ATR(14)
RR_TP1 = 1.0                # TP1 = 1R
RR_TP2 = 2.0                # TP2 = 2R (matches a 1:2 risk/reward)


# ---------- indicators (all expect oldest -> newest order) ----------
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
    # Wilder's smoothing
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

    if rsi_val is not None:
        if rsi_val < 30:
            score += 1
            notes.append(f"RSI {rsi_val:.1f}: oversold")
        elif rsi_val > 70:
            score -= 1
            notes.append(f"RSI {rsi_val:.1f}: overbought")
        else:
            notes.append(f"RSI {rsi_val:.1f}: neutral")

    if macd_val:
        if macd_val["macd"] > macd_val["signal"]:
            score += 1
            notes.append("MACD: bullish crossover")
        else:
            score -= 1
            notes.append("MACD: bearish crossover")

    if score >= 2:
        label, emoji, direction = "STRONG BUY", "🟢🟢", "buy"
    elif score == 1:
        label, emoji, direction = "BUY", "🟢", "buy"
    elif score == -1:
        label, emoji, direction = "SELL", "🔴", "sell"
    elif score <= -2:
        label, emoji, direction = "STRONG SELL", "🔴🔴", "sell"
    else:
        label, emoji, direction = "NEUTRAL", "⚪", None

    plan = None
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

    return {
        "price": price, "label": label, "emoji": emoji, "notes": notes,
        "score": score, "direction": direction, "plan": plan,
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
    values = list(reversed(data["values"]))  # oldest -> newest
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


def main():
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    strong_blocks = []
    quiet_lines = []

    for symbol, label in PAIRS:
        try:
            highs, lows, closes = fetch_ohlc(symbol, api_key)
            sig = build_signal(highs, lows, closes)
            d = decimals_for(symbol)

            if "STRONG" in sig["label"] and sig["plan"]:
                p = sig["plan"]
                rr_note = f"1 : {RR_TP2:.0f}"
                block = (
                    f"{sig['emoji']} <b>{label}</b> — <b>{sig['label']}</b>\n"
                    f"Entry: <code>{p['entry']:.{d}f}</code>\n"
                    f"SL: <code>{p['sl']:.{d}f}</code>\n"
                    f"TP1: <code>{p['tp1']:.{d}f}</code>  (1R)\n"
                    f"TP2: <code>{p['tp2']:.{d}f}</code>  (2R)\n"
                    f"Risk:Reward  {rr_note}\n"
                    f"<i>{' · '.join(sig['notes'])}</i>"
                )
                strong_blocks.append(block)
            else:
                quiet_lines.append(f"{sig['emoji']} {label}: {sig['label']}")
        except Exception as e:
            quiet_lines.append(f"⚠️ {label}: error — {e}")
        time.sleep(REQUEST_GAP_SECONDS)

    if strong_blocks:
        header = "<b>🎯 High-Conviction Setups</b>\n\n"
        message = header + "\n\n".join(strong_blocks)
        if quiet_lines:
            message += "\n\n<i>No strong setup: " + "; ".join(quiet_lines) + "</i>"
    else:
        message = "<b>📊 Forex & Gold Scan</b>\n\nNo high-conviction setups this cycle.\n\n" + "\n".join(quiet_lines)

    send_telegram(tg_token, tg_chat, message)
    print(message)


if __name__ == "__main__":
    main()
