import os
import time
import json
import logging
import anthropic
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
POLYGON_API_KEY  = os.getenv("POLYGON_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TICKERS          = ["AAPL", "GOOGL", "AMZN", "MSFT"]
POLL_INTERVAL    = 10   # seconds between full cycles
FETCH_DELAY      = 13   # seconds between each ticker fetch (free tier = 5 calls/min)
MAX_RETRIES      = 3    # Polygon fetch retries
RETRY_BACKOFF    = 15   # seconds to wait after a 429 rate-limit error

# ── Logging (errors go to file, NOT terminal) ─────────────────────────────────
logging.basicConfig(
    filename="agent_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Polygon: fetch with retry + backoff ───────────────────────────────────────
def fetch_ohlcv(ticker: str) -> dict | None:
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?apiKey={POLYGON_API_KEY}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            results = r.json().get("results", [])
            if results:
                return results[0]
            # API responded but no data (e.g. market closed)
            logging.error(f"{ticker}: Empty results on attempt {attempt}")
        except requests.exceptions.Timeout:
            logging.error(f"{ticker}: Timeout on attempt {attempt}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            logging.error(f"{ticker}: HTTP {status_code} on attempt {attempt}")
            if status_code == 429:
                time.sleep(RETRY_BACKOFF)  # Wait longer on rate limit, then retry
            else:
                break  # Don't retry on other 4xx (bad key, invalid ticker)
        except Exception as e:
            logging.error(f"{ticker}: Unexpected error on attempt {attempt}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_BACKOFF)

    return None  # All retries exhausted

# ── Generate commentary from OHLCV ───────────────────────────────────────────
def generate_commentary(ticker: str, data: dict) -> str:
    o, h, l, c, v = data["o"], data["h"], data["l"], data["c"], data["v"]
    change    = c - o
    pct       = (change / o) * 100
    direction = "rose" if change >= 0 else "fell"
    volume_m  = v / 1_000_000
    return (
        f"{ticker} {direction} {abs(pct):.2f}% today. "
        f"It opened at ${o:.2f}, reached a high of ${h:.2f}, "
        f"a low of ${l:.2f}, and closed at ${c:.2f}. "
        f"Volume was {volume_m:.1f}M shares."
    )

# ── Safe JSON extraction (handles markdown fences) ───────────────────────────
def extract_json(text: str) -> dict | None:
    # Strip ```json ... ``` or ``` ... ``` fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None

SENTIMENT_PROMPT = """You are a financial sentiment analyzer.

Analyze the sentiment of this stock market commentary.
Respond with ONLY a raw JSON object — no explanation, no markdown, no code fences.

Format exactly:
{{"sentiment": "Positive" | "Negative" | "Neutral", "confidence": 0.0-1.0, "reason": "one short sentence"}}

Commentary: {commentary}"""

# ── Claude sentiment: temperature=0 + retry on bad JSON ──────────────────────
def analyze_sentiment(commentary: str) -> dict:
    fallback = {"sentiment": "Neutral", "confidence": 0.0, "reason": "Analysis unavailable."}

    for attempt in range(1, 3):  # max 2 attempts
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=150,
                temperature=0,          # deterministic output — no random flipping
                messages=[{
                    "role": "user",
                    "content": SENTIMENT_PROMPT.format(commentary=commentary)
                }]
            )
            raw  = message.content[0].text
            data = extract_json(raw)

            if data and "sentiment" in data and "confidence" in data:
                # Clamp confidence to valid range
                data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
                return data

            logging.error(f"Bad JSON on attempt {attempt}: {raw}")

        except Exception as e:
            logging.error(f"Claude API error on attempt {attempt}: {e}")
            break

    return fallback

# ── Terminal display ───────────────────────────────────────────────────────────
SENTIMENT_COLOR = {
    "Positive": "\033[92m",
    "Negative": "\033[91m",
    "Neutral":  "\033[93m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

def print_dashboard(results: list[dict], cycle: int):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{BOLD}{'═'*62}{RESET}")
    print(f"{BOLD}  📈 Stock Sentiment Agent  |  Cycle #{cycle}  |  {now}{RESET}")
    print(f"{BOLD}{'═'*62}{RESET}")

    for r in results:
        ticker    = r["ticker"]
        sentiment = r["sentiment"]
        conf      = r["confidence"]
        reason    = r["reason"]
        commentary= r["commentary"]
        color     = SENTIMENT_COLOR.get(sentiment, DIM)
        status    = r.get("status", "ok")

        print(f"\n  {BOLD}{ticker}{RESET}", end="")
        if status != "ok":
            print(f"  {DIM}[{status}]{RESET}", end="")
        print()

        print(f"  {DIM}{commentary}{RESET}")
        print(f"  Sentiment : {color}{BOLD}{sentiment}{RESET}  "
              f"(confidence: {conf:.0%})")
        print(f"  Reason    : {reason}")

    print(f"\n{BOLD}{'─'*62}{RESET}")
    print(f"  {DIM}Next update in {POLL_INTERVAL}s  |  "
          f"Errors logged to agent_errors.log  |  Ctrl+C to stop{RESET}\n")

# ── Structured fallback result ────────────────────────────────────────────────
def make_fallback(ticker: str, reason: str) -> dict:
    return {
        "ticker":     ticker,
        "commentary": "Data unavailable.",
        "sentiment":  "Neutral",
        "confidence": 0.0,
        "reason":     reason,
        "status":     "fetch_failed",
    }

# ── Main agent loop ────────────────────────────────────────────────────────────
def run_agent():
    print(f"{BOLD}Starting Stock Sentiment Agent...{RESET}")
    print(f"Monitoring : {', '.join(TICKERS)}")
    print(f"Interval   : {POLL_INTERVAL}s")
    print(f"Error log  : agent_errors.log\n")

    cycle = 0
    while True:
        cycle += 1
        results = []

        for i, ticker in enumerate(TICKERS):
            if i > 0:
                time.sleep(FETCH_DELAY)  # Respect Polygon free-tier rate limit
            data = fetch_ohlcv(ticker)

            if data is None:
                results.append(make_fallback(ticker, "Polygon returned no data after retries."))
                continue

            commentary = generate_commentary(ticker, data)
            analysis   = analyze_sentiment(commentary)

            results.append({
                "ticker":     ticker,
                "commentary": commentary,
                "sentiment":  analysis["sentiment"],
                "confidence": analysis["confidence"],
                "reason":     analysis["reason"],
                "status":     "ok",
            })

        print_dashboard(results, cycle)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    try:
        run_agent()
    except KeyboardInterrupt:
        print(f"\n{BOLD}Agent stopped.{RESET}")