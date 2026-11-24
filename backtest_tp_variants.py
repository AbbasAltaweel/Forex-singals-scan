"""
TP1 / win-rate variants test. Uses the SAME entry signals and stops as the live bots
(imported from backtest_long_history_td.py) and only changes where TP1/TP2 sit.

Model (consistent R accounting, spread subtracted):
  SL before TP1                    = -1R
  TP1 hit (fraction f closed), then stop moved to entry:
      stopped at entry             = f*TP1
      TP2 hit                      = f*TP1 + (1-f)*TP2
Same-candle ambiguity is resolved against us (SL / entry-stop first).
"win rate" is shown two ways: TP1-hit % (what the bot would call a win) and
the money result, which is what actually matters.
"""
import os, sys, html, datetime
import backtest_long_history_td as bt

# name, tp1_R, tp2_R, fraction closed at TP1
VARIANTS = [
    ("Current 1.0/2.0 half",   1.0,  2.0, 0.5),
    ("Close TP1 0.5, TP2 2.0", 0.5,  2.0, 0.5),
    ("Close TP1 0.33, TP2 2.0", 0.33, 2.0, 0.5),
    ("TP1 0.5 bank 70%, TP2 2", 0.5,  2.0, 0.7),
    ("Single TP 0.5 (no runner)", 0.5, 0.5, 1.0),
]


def sim_trade(direction, entry, sl, highs, lows, start, tp1_r, tp2_r, frac, max_bars=400):
    risk = abs(entry - sl)
    sgn = 1 if direction == "buy" else -1
    tp1 = entry + sgn * risk * tp1_r
    tp2 = entry + sgn * risk * tp2_r
    tp1_hit = False
    n = len(highs)
    for i in range(start, min(n, start + max_bars)):
        hi, lo = highs[i], lows[i]
        hit_sl = (lo <= sl) if sgn == 1 else (hi >= sl)
        hit_entry = (lo <= entry) if sgn == 1 else (hi >= entry)
        hit1 = (hi >= tp1) if sgn == 1 else (lo <= tp1)
        hit2 = (hi >= tp2) if sgn == 1 else (lo <= tp2)
        if not tp1_hit:
            if hit_sl:
                return i, "sl", -1.0, risk
            if hit1:
                tp1_hit = True
                if frac >= 1.0:
                    return i, "tp1", tp1_r, risk
            continue
        if hit_entry:
            return i, "be", frac * tp1_r, risk
        if hit2:
            return i, "tp2", frac * tp1_r + (1 - frac) * tp2_r, risk
    return None


def run(symbol, d, variant):
    _, tp1_r, tp2_r, frac = variant
    out, i, n = [], 60, len(d["closes"])
    while i < n - 1:
        lo = max(0, i - bt.WINDOW + 1)
        sig = bt.build_signal(d["highs"][lo:i + 1], d["lows"][lo:i + 1], d["closes"][lo:i + 1])
        if sig["direction"] and sig["plan"] and sig["trade_type"] == "Swing":
            p = sig["plan"]
            res = sim_trade(sig["direction"], p["entry"], p["sl"], d["highs"], d["lows"],
                            i + 1, tp1_r, tp2_r, frac)
            if res is None:
                i += 1
                continue
            j, kind, r, risk = res
            spread_r = bt.SPREADS.get(symbol, 0.0) / risk if risk else 0
            out.append({"kind": kind, "r": r - spread_r,
                        "year": datetime.datetime.utcfromtimestamp(d["times"][j]).year})
            i = j + 1
        else:
            i += 1
    return out


def summarize(trades):
    n = len(trades)
    if not n:
        return None
    tp1_rate = sum(1 for t in trades if t["kind"] != "sl") / n * 100
    total = sum(t["r"] for t in trades)
    gains = sum(t["r"] for t in trades if t["r"] > 0)
    losses = abs(sum(t["r"] for t in trades if t["r"] < 0))
    pf = gains / losses if losses else float("inf")
    years = {}
    for t in trades:
        years[t["year"]] = years.get(t["year"], 0.0) + t["r"]
    posy = sum(1 for v in years.values() if v > 0)
    return {"n": n, "tp1": tp1_rate, "total": total, "pf": pf, "avg": total / n,
            "posy": posy, "ny": len(years)}


def build_report(results, errors):
    lines = ["<b>TP1 variants (same signals, same stops)</b>",
             "TP1% = trades that reach the first target. R = money result after spread.", ""]
    for tf in ("H1", "H4"):
        lines.append(f"<b>{tf}, all pairs combined</b>")
        rows = ["Variant                   N   TP1%   TotalR  PF   +Yrs"]
        for v in VARIANTS:
            tr = results.get((tf, v[0]), [])
            s = summarize(tr)
            if not s:
                rows.append(f"{v[0][:24]:<24} no trades")
                continue
            rows.append(f"{v[0][:24]:<24} {s['n']:>5} {s['tp1']:>4.0f}% {s['total']:>+7.0f} {s['pf']:>4.2f} {s['posy']}/{s['ny']}")
        lines.append("<pre>" + "\n".join(rows) + "</pre>")
    best = None
    for tf in ("H1", "H4"):
        for v in VARIANTS:
            s = summarize(results.get((tf, v[0]), []))
            if s and (best is None or s["total"] > best[0]):
                best = (s["total"], tf, v[0], s)
    if best and best[0] > 2 * (best[3]["n"] ** 0.5) and best[3]["posy"] / best[3]["ny"] >= 0.6:
        lines.append(f"Best: {tf_safe(best)} is positive beyond normal noise, but forward-test it before trusting it.")
    else:
        lines.append("<b>Verdict:</b> no variant is profitable across years. A closer TP1 raises the win rate "
                     "but does not create an edge.")
    if errors:
        lines.append("Errors: " + "; ".join(html.escape(e) for e in errors))
    return "\n".join(lines)


def tf_safe(best):
    return html.escape(f"{best[2]} on {best[1]} ({best[0]:+.0f}R)")


def main():
    key = "".join(os.environ["TWELVE_DATA_API_KEY"].split())
    tg_token, tg_chat = os.environ["TELEGRAM_BOT_TOKEN"].strip(), os.environ["TELEGRAM_CHAT_ID"].strip()
    results, errors = {}, []
    for _inst, symbol, label in bt.PAIRS:
        for tf, interval in (("H1", "1h"), ("H4", "4h")):
            try:
                t, h, l, c = bt.fetch_history(symbol, key, interval)
                d = {"times": t, "highs": h, "lows": l, "closes": c}
                for v in VARIANTS:
                    results.setdefault((tf, v[0]), []).extend(run(symbol, d, v))
                print(f"{label} {tf} done")
            except Exception as e:
                errors.append(f"{label} {tf}: {str(e)[:80]}")
    msg = build_report(results, errors)
    print(msg)
    bt.send_telegram(tg_token, tg_chat, msg)


if __name__ == "__main__":
    main()
