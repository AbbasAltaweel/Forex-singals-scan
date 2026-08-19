"""
Forex + Gold signal scanner -> Telegram alerts.

Runs once per invocation (designed to be triggered on a schedule, e.g. by
GitHub Actions every 15 minutes). Each run:
  1. Pulls the latest hourly candles for each pair from Twelve Data.
  2. Computes SMA20/50, RSI(14), and MACD(12,26,9) on closing prices.
  3. Combines them into one signal per pair.
  4. Sends a single summary message to your Telegram chat.

Required environment variables (set as GitHub Actions secrets):
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

REQUEST_GAP_SECONDS = 7  # stay under Twelve Data's free-tier rate limit


def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[:period]) / period


def ema_series(values_chronological, period):
    k = 2 / (period + 1)
    e = sum(values_chronological[:period]) / period
    out = [e]
    for v in values_chronological[period:]:
        e = v * k + e * (1 - k)
        out.append(e)
    return out


def rsi(values, period=14):
    if len(values) < period + 1:
        return None
    chron = list(reversed(values))
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        diff = chron[i] - chron[i - 1]
        if diff >= 0:
            gains += diff
        else:
            losses -= diff
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(chron)):
        diff = chron[i] - chron[i - 1]
        gain = diff if diff > 0 else 0
        loss = -diff if diff < 0 else 0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(values):
    chron = list(reversed(values))
    if len(chron) < 35:
        return None
    ema12 = ema_series(chron, 12)
    ema26 = ema_series(chron, 26)
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


def build_signal(closes):
    price = closes[0]
    sma20, sma50 = sma(closes, 20), sma(closes, 50)
    rsi_val = rsi(closes, 14)
    macd_val = macd(closes)

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
        label, emoji = "STRONG BUY", "🟢🟢"
    elif score == 1:
        label, emoji = "BUY", "🟢"
    elif score == -1:
        label, emoji = "SELL", "🔴"
    elif score <= -2:
        label, emoji = "STRONG SELL", "🔴🔴"
    else:
        label, emoji = "NEUTRAL", "⚪"

    return {"price": price, "label": label, "emoji": emoji, "notes": notes, "score": score}


def fetch_closes(symbol, api_key):
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": "1h", "outputsize": 100, "apikey": api_key}
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if data.get("status") == "error" or "code" in data:
        raise RuntimeError(data.get("message", "API error"))
    if "values" not in data:
        raise RuntimeError("No data returned")
    return [float(v["close"]) for v in data["values"]]


def send_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=20)
    if r.status_code != 200:
        print(f"Telegram send failed: {r.status_code} {r.text}", file=sys.stderr)


def main():
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    lines = ["<b>📊 Forex &amp; Gold Signal Scan</b>", ""]
    strong_flags = []

    for symbol, label in PAIRS:
        try:
            closes = fetch_closes(symbol, api_key)
            sig = build_signal(closes)
            decimals = 3 if symbol == "USD/JPY" else 2 if symbol == "XAU/USD" else 5
            price_str = f"{sig['price']:.{decimals}f}"
            lines.append(f"{sig['emoji']} <b>{label}</b>  {price_str}  —  {sig['label']}")
            if "STRONG" in sig["label"]:
                strong_flags.append(f"{label}: {sig['label']}")
        except Exception as e:
            lines.append(f"⚠️ <b>{label}</b> — error: {e}")
        time.sleep(REQUEST_GAP_SECONDS)

    if strong_flags:
        lines.insert(2, "🔥 <b>Strong signals:</b> " + "; ".join(strong_flags))
        lines.insert(3, "")

    message = "\n".join(lines)
    send_telegram(tg_token, tg_chat, message)
    print(message)


if __name__ == "__main__":
    main()
