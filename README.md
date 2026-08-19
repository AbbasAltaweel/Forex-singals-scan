# Forex + Gold Signal Scanner → Telegram

Scans EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD and
XAU/USD (gold) every 15 minutes and sends a signal summary to your Telegram,
using GitHub Actions as the free scheduler — nothing needs to stay running
on your own computer.

Signals are a composite of SMA20/50 trend, RSI(14), and MACD(12,26,9) —
same method as the browser scanner, just automated.

## Setup (10 minutes, one time)

### 1. Create a Telegram bot
1. Open Telegram, search for **@BotFather**, start a chat.
2. Send `/newbot`, follow the prompts (choose a name and username).
3. BotFather gives you a **bot token** — looks like `123456:ABC-DEF...`. Save it.
4. Start a chat with your new bot (search its username, hit Start) and send it any message — this lets it message you back.

### 2. Get your chat ID
1. Visit this URL in your browser, replacing `<TOKEN>` with your bot token:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
2. Find `"chat":{"id":123456789,...}` in the response — that number is your **chat ID**.
   (If you see nothing, make sure you've sent the bot a message first, then refresh.)

### 3. Get a free Twelve Data API key
1. Sign up at https://twelvedata.com (free, no card required).
2. Copy your API key from the dashboard.

### 4. Create a GitHub repo with these files
1. Create a new **private** repo on GitHub (private keeps your setup out of public view; the secrets are hidden either way).
2. Upload these three files, keeping the folder structure:
   - `scanner.py`
   - `requirements.txt`
   - `.github/workflows/scan.yml`

### 5. Add your secrets
In the repo: **Settings → Secrets and variables → Actions → New repository secret**. Add all three:
- `TWELVE_DATA_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### 6. Turn it on
- Go to the **Actions** tab of your repo → you should see "Forex Signal Scan".
- Click into it and hit **"Run workflow"** to test it manually first — check Telegram for the message.
- If it works, it'll now run automatically every 15 minutes.

## Notes & limits

- **Rate limit**: Twelve Data's free tier allows 800 requests/day. Scanning 8 symbols every 15 minutes uses ~768/day — right at the edge. If you add more pairs, either reduce frequency or upgrade the API plan.
- **GitHub Actions free minutes**: scheduled workflows on public repos are free and unlimited; private repos get 2,000 free minutes/month, and this job only takes ~1 minute per run (~2,900 min/month at 15-min intervals) — so a **public** repo is the safer free option, or reduce frequency to every 30 min for a private repo to stay under the limit.
- **Inactivity pause**: GitHub automatically disables scheduled workflows after 60 days with no repo activity. Any commit (even a trivial one) resets that clock.
- **Not financial advice** — this is a rules-based technical signal, not a guarantee. Always confirm before trading.
