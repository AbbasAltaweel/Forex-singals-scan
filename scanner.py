"""
Forex + Gold signal scanner with trade tracking, auto win-rate stats,
and quiet messaging -> Telegram alerts.

Behavior:
  - Opens a trade only on full 3/3 indicator agreement (trend + RSI + MACD).
  - Tags each trade Scalp or Swing based on target size relative to price.
  - Every run, re-checks open trades' latest candle against levels:
      TP1 hit -> partial profit + move stop to breakeven (trade stays open)
      TP2 hit -> full win, closed
      SL hit (before TP1) -> loss, closed
      Breakeven hit (after TP1) -> closed flat, partial gain already banked
  - Every closed trade is logged to history with an approximate R result:
      TP2 = +2R, SL = -1R, breakeven-after-TP1 = +0.5R (assumes half
      position closed at TP1, remainder stopped at breakeven).
  - Once every 24h, sends a summary: trades closed, win rate, total R.
  - QUIET MODE: if nothing happened this cycle (no update, no new signal,
    no summary due), no Telegram message is sent at all -- only real
    changes reach you.

Required environment variables (GitHub Actions secrets):
  TWELVE_DATA_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
import time
import json
import datetime
from zoneinfo import ZoneInfo
import requests

TZ = ZoneInfo("America/Toronto")

SESSION_OPENS = [
    ("Sydney", 17, "Asian session (Sydney) opening — week's liquidity begins."),
    ("Tokyo", 19, "Tokyo session opening, overlapping with Sydney."),
    ("London", 3, "London session opening — European liquidity, often a volatility pickup."),
    ("New York", 8, "New York session opening — overlaps London, typically the highest volume window."),
]

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
SCALP_PCT_THRESHOLD = 0.4

R_TP2 = 2.0
R_SL = -1.0
R_BREAKEVEN_AFTER_TP1 = 0.5  # approximation: half position banked at TP1


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
        # momentum confirmation (not reversal) -- keeps RSI aligned with
        # trend/MACD instead of contradicting them
        if rsi_val > 50:
            score += 1
            notes.append(f"RSI {rsi_val:.1f}: bullish momentum")
        else:
            score -= 1
            notes.append(f"RSI {rsi_val:.1f}: bearish momentum")
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


HEARTBEAT_SECONDS = 8 * 60 * 60  # send a status ping at least this often


RANKING_FILE = "pair_ranking.json"


def now_local():
    return datetime.datetime.now(TZ)


def is_market_open(dt):
    """Forex convention: closed Fri 5pm ET through Sun 5pm ET."""
    weekday = dt.weekday()  # Mon=0 ... Sun=6
    if weekday == 4 and dt.hour >= 17:
        return False
    if weekday == 5:
        return False
    if weekday == 6 and dt.hour < 17:
        return False
    return True


def load_top_pairs():
    """Reads the ranking produced by the last backtest run.
    Returns (allowed_set, best_symbol_or_None)."""
    if os.path.exists(RANKING_FILE):
        try:
            with open(RANKING_FILE, "r") as f:
                data = json.load(f)
                ranked = data.get("ranked_symbols", [])
                if ranked:
                    return set(ranked[:3]), ranked[0]
        except Exception:
            pass
    return set(), None


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                data.setdefault("trades", {})
                data.setdefault("closed", [])
                data.setdefault("last_summary_date", None)
                data.setdefault("last_heartbeat", 0)
                data.setdefault("session_notices", {})
                data.setdefault("friday_notice_date", None)
                data.setdefault("sunday_prep_date", None)
                return data
        except Exception:
            pass
    return {"trades": {}, "closed": [], "last_summary_date": None, "last_heartbeat": 0,
            "session_notices": {}, "friday_notice_date": None, "sunday_prep_date": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_open_trade(trade, highs, lows, closes, d):
    last_high, last_low = highs[-1], lows[-1]
    direction = trade["direction"]
    entry, sl, tp1, tp2 = trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]
    pair_label = trade["label"]
    risk_dist = trade.get("risk_dist", abs(entry - sl))

    if not trade.get("tp1_hit"):
        if direction == "buy":
            hit_tp1, hit_sl = last_high >= tp1, last_low <= sl
        else:
            hit_tp1, hit_sl = last_low <= tp1, last_high >= sl

        if hit_sl:
            msg = (f"🔴 <b>{pair_label}</b> — SL hit\n"
                   f"Closed at <code>{sl:.{d}f}</code>. <i>Result: -1R</i>")
            return msg, trade, False, ("sl", R_SL)
        if hit_tp1:
            trade["tp1_hit"] = True
            trade["effective_sl"] = entry
            msg = (f"🟡 <b>{pair_label}</b> — TP1 hit\n"
                   f"Bank partial profit, move stop to breakeven (<code>{entry:.{d}f}</code>). "
                   f"Holding rest for TP2 <code>{tp2:.{d}f}</code>.")
            return msg, trade, True, None
        return None, trade, True, None
    else:
        effective_sl = trade.get("effective_sl", entry)

        # trailing checkpoint: once price pushes to 1.5R, lock in more profit
        # by trailing the stop to 0.75R (only fires once)
        if not trade.get("trail_hit"):
            trail_level = entry + risk_dist * 1.5 if direction == "buy" else entry - risk_dist * 1.5
            hit_trail = (last_high >= trail_level) if direction == "buy" else (last_low <= trail_level)
            if hit_trail:
                trade["trail_hit"] = True
                new_sl = entry + risk_dist * 0.75 if direction == "buy" else entry - risk_dist * 0.75
                trade["effective_sl"] = new_sl
                msg = (f"🔵 <b>{pair_label}</b> — Trailing update\n"
                       f"Price extended toward TP2. Move stop to <code>{new_sl:.{d}f}</code> "
                       f"(locks in +0.75R). Still holding for TP2 <code>{tp2:.{d}f}</code>.")
                return msg, trade, True, None

        effective_sl = trade.get("effective_sl", entry)
        hit_tp2 = (last_high >= tp2) if direction == "buy" else (last_low <= tp2)
        hit_be = (last_low <= effective_sl) if direction == "buy" else (last_high >= effective_sl)

        if hit_tp2:
            msg = (f"🟢🟢 <b>{pair_label}</b> — TP2 hit! Full target\n"
                   f"Closed at <code>{tp2:.{d}f}</code>. <i>Result: +2R</i>")
            return msg, trade, False, ("tp2", R_TP2)
        if hit_be:
            locked = "+0.75R (trailed)" if trade.get("trail_hit") else "+0.5R (TP1 banked)"
            msg = (f"⚪ <b>{pair_label}</b> — Stopped out at trailed level\n"
                   f"Closed at <code>{effective_sl:.{d}f}</code>. <i>Result: {locked}</i>")
            r_val = 0.75 if trade.get("trail_hit") else R_BREAKEVEN_AFTER_TP1
            return msg, trade, False, ("breakeven", r_val)
        return None, trade, True, None


def build_summary(closed_trades):
    if not closed_trades:
        return None
    wins = [t for t in closed_trades if t["result"] in ("tp2", "breakeven")]
    losses = [t for t in closed_trades if t["result"] == "sl"]
    total_r = sum(t["r"] for t in closed_trades)
    win_rate = (len(wins) / len(closed_trades)) * 100 if closed_trades else 0
    return (
        f"<b>📈 24h Summary</b>\n\n"
        f"Trades closed: {len(closed_trades)}\n"
        f"Wins: {len(wins)}  ·  Losses: {len(losses)}\n"
        f"Win rate: {win_rate:.0f}%\n"
        f"Total R: {'+' if total_r >= 0 else ''}{total_r:.2f}\n\n"
        f"<i>Breakeven-after-TP1 counted as +0.5R (assumes half position banked at TP1).</i>"
    )


def build_heartbeat(trades):
    now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if not trades:
        return (f"<b>✅ Status check-in</b>  ({now_str})\n\n"
                f"Bot is running. No open trades right now — watching all {len(PAIRS)} pairs for a new full-agreement setup.")
    lines = []
    for symbol, t in trades.items():
        stage = "past TP1, targeting TP2 (risk-free)" if t.get("tp1_hit") else "targeting TP1"
        lines.append(f"• {t['label']} — {t['direction'].upper()} [{t.get('trade_type','?')}] — {stage}")
    return (f"<b>✅ Status check-in</b>  ({now_str})\n\n"
            f"Bot is running. Open trades ({len(trades)}):\n" + "\n".join(lines))


def build_friday_warning(trades):
    lines = []
    for symbol, t in trades.items():
        if t.get("tp1_hit"):
            advice = "TP1 already banked, stop at breakeven or better — safe to hold over the weekend (no downside risk left)."
        else:
            advice = "TP1 not yet hit — your stop is still exposing you to weekend gap risk. Consider closing manually or tightening your stop before the close."
        lines.append(f"• {t['label']} ({t['direction'].upper()}) — {advice}")
    if not lines:
        lines = ["No open trades right now — nothing to decide on."]
    return f"<b>🕔 Market closes in ~1 hour</b> (5pm ET Friday)\n\n" + "\n".join(lines)


def build_sunday_prep(api_key):
    lines = []
    for symbol, label in PAIRS:
        try:
            highs, lows, closes = fetch_ohlc(symbol, api_key)
            sig = build_signal(highs, lows, closes)
            bias = "Bullish bias" if sig["direction"] == "buy" else "Bearish bias" if sig["direction"] == "sell" else "No clear bias"
            lines.append(f"{label}: {bias}")
        except Exception as e:
            lines.append(f"{label}: data unavailable ({e})")
        time.sleep(REQUEST_GAP_SECONDS)
    return ("<b>🗓️ Sunday Market Prep</b>\n\nTechnical bias heading into the week (based on last available candles):\n"
            + "\n".join(lines) +
            "\n\n<i>Technical read only — no news/economic calendar included yet.</i>")


def main():
    api_key = os.environ["TWELVE_DATA_API_KEY"]
    tg_token = os.environ["TELEGRAM_BOT_TOKEN"]
    tg_chat = os.environ["TELEGRAM_CHAT_ID"]

    state = load_state()
    trades = state["trades"]
    closed_history = state["closed"]
    top_pairs, best_pair = load_top_pairs()

    now = now_local()
    today_str = now.date().isoformat()
    market_open = is_market_open(now)

    weekend_parts = []

    # Friday close warning (fires once, while market is still open)
    if now.weekday() == 4 and now.hour == 16 and state.get("friday_notice_date") != today_str:
        weekend_parts.append(build_friday_warning(trades))
        state["friday_notice_date"] = today_str

    # Sunday prep (fires once, while market still closed, ~2h before open)
    if now.weekday() == 6 and now.hour == 15 and state.get("sunday_prep_date") != today_str:
        weekend_parts.append(build_sunday_prep(api_key))
        state["sunday_prep_date"] = today_str

    update_blocks = []
    new_signal_blocks = []
    session_blocks = []
    summary_block = None
    heartbeat_block = None

    if market_open:
        # session-open pings
        for name, hour, desc in SESSION_OPENS:
            key = f"{name}_{today_str}"
            if now.hour == hour and state["session_notices"].get(key) is not True:
                session_blocks.append(f"<b>🌍 {name} session opening</b>\n{desc}")
                state["session_notices"][key] = True

        for symbol, label in PAIRS:
            try:
                highs, lows, closes = fetch_ohlc(symbol, api_key)
                d = decimals_for(symbol)

                if symbol in trades:
                    msg, updated_trade, still_open, result = check_open_trade(trades[symbol], highs, lows, closes, d)
                    if msg:
                        update_blocks.append(msg)
                    if still_open:
                        trades[symbol] = updated_trade
                    elif result:
                        result_type, r_value = result
                        record = {
                            "pair": label, "direction": updated_trade["direction"],
                            "trade_type": updated_trade.get("trade_type"),
                            "result": result_type, "r": r_value,
                            "closed_at": int(time.time()),
                        }
                        closed_history.append(record)
                        del trades[symbol]
                else:
                    # only take new trades on top-ranked pairs (falls back to
                    # all pairs if no backtest ranking has been generated yet)
                    eligible = (not top_pairs) or (symbol in top_pairs)
                    sig = build_signal(highs, lows, closes)
                    if eligible and sig["direction"] and sig["plan"] and sig["trade_type"] == "Swing":
                        p = sig["plan"]
                        trade = {
                            "label": label, "direction": sig["direction"],
                            "entry": p["entry"], "sl": p["sl"], "tp1": p["tp1"], "tp2": p["tp2"],
                            "tp1_hit": False, "trade_type": sig["trade_type"],
                            "opened_at": int(time.time()),
                            "risk_dist": abs(p["entry"] - p["sl"]),
                        }
                        trades[symbol] = trade
                        dir_word = "BUY" if sig["direction"] == "buy" else "SELL"
                        emoji = "🟢🟢" if sig["direction"] == "buy" else "🔴🔴"
                        star = " ⭐ (top-ranked pair)" if symbol == best_pair else ""
                        order_type = "BUY NOW" if sig["direction"] == "buy" else "SELL NOW"
                        limit_type = "Buy Limit" if sig["direction"] == "buy" else "Sell Limit"
                        buffer_pips = trade["risk_dist"] * 0.2  # ~20% of risk distance
                        block = (
                            f"{emoji} <b>{label}</b> — <b>{dir_word}</b>  [{sig['trade_type']}]{star}\n"
                            f"Order type: <b>{order_type}</b> (market)\n"
                            f"Entry: <code>{p['entry']:.{d}f}</code>\n"
                            f"SL: <code>{p['sl']:.{d}f}</code>\n"
                            f"TP1: <code>{p['tp1']:.{d}f}</code>  (1R)\n"
                            f"TP2: <code>{p['tp2']:.{d}f}</code>  (2R)\n"
                            f"Risk:Reward  1 : 2\n"
                            f"<i>Full agreement: {' · '.join(sig['notes'])}</i>\n"
                            f"<i>If price has already moved more than ~{buffer_pips:.{d}f} away from entry by the time you act, "
                            f"place a <b>{limit_type}</b> at <code>{p['entry']:.{d}f}</code> instead of chasing at market.</i>"
                        )
                        new_signal_blocks.append(block)
            except Exception as e:
                print(f"Error on {label}: {e}", file=sys.stderr)
            time.sleep(REQUEST_GAP_SECONDS)

        # daily summary check (once per UTC day)
        today_utc = datetime.date.today().isoformat()
        if state.get("last_summary_date") != today_utc:
            cutoff = int(time.time()) - 86400
            recent_closed = [t for t in closed_history if t["closed_at"] >= cutoff]
            summary_block = build_summary(recent_closed)
            state["last_summary_date"] = today_utc

        # heartbeat (only during market hours -- no weekend pings)
        now_ts = int(time.time())
        if now_ts - state.get("last_heartbeat", 0) >= HEARTBEAT_SECONDS:
            heartbeat_block = build_heartbeat(trades)
            state["last_heartbeat"] = now_ts  # only reset the clock when we actually send one
    else:
        print("Market closed — skipping scan (weekend quiet mode).")

    state["trades"] = trades
    state["closed"] = closed_history
    save_state(state)

    parts = list(weekend_parts) + session_blocks
    if update_blocks:
        parts.append("<b>📌 Trade Updates</b>\n\n" + "\n\n".join(update_blocks))
    if new_signal_blocks:
        parts.append("<b>🎯 New High-Conviction Setups</b>\n\n" + "\n\n".join(new_signal_blocks))
    if summary_block:
        parts.append(summary_block)
    if heartbeat_block and len(parts) == 0:
        parts.append(heartbeat_block)

    if parts:
        message = "\n\n---\n\n".join(parts)
        send_telegram(tg_token, tg_chat, message)
        print(message)
    else:
        print("Nothing to report this cycle — no message sent.")


if __name__ == "__main__":
    main()
