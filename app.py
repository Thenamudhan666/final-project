"""
Flask API server for the Stock Sentiment Agent.
Exposes endpoints consumed by the React frontend.
"""

import time
import threading
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS

from InfoFetchAgent import (
    fetch_ohlcv,
    generate_commentary,
    analyze_sentiment,
    TICKERS,
    FETCH_DELAY,
)

app = Flask(__name__)
CORS(app)

# Global cache for the latest results
LATEST_DATA = {
    "timestamp": None,
    "tickers": [],
    "is_fetching": True  # True during the very first fetch
}

def background_fetch_loop():
    """Background thread that continuously fetches data and updates the cache."""
    print("Background fetch thread started...")
    while True:
        results = []
        for i, ticker in enumerate(TICKERS):
            if i > 0:
                print(f"Waiting {FETCH_DELAY}s for rate limit...")
                time.sleep(FETCH_DELAY)  # respect Polygon free-tier rate limit

            print(f"Fetching data for {ticker}...")
            data = fetch_ohlcv(ticker)

            if data is None:
                results.append({
                    "ticker": ticker,
                    "commentary": "Data unavailable.",
                    "sentiment": "Neutral",
                    "confidence": 0.0,
                    "reason": "Polygon returned no data after retries.",
                    "status": "fetch_failed",
                })
                continue

            commentary = generate_commentary(ticker, data)
            analysis = analyze_sentiment(commentary)

            o, h, l, c, v = data["o"], data["h"], data["l"], data["c"], data["v"]
            change_pct = ((c - o) / o) * 100 if o else 0

            results.append({
                "ticker": ticker,
                "commentary": commentary,
                "sentiment": analysis["sentiment"],
                "confidence": analysis["confidence"],
                "reason": analysis["reason"],
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "change_pct": round(change_pct, 2),
                "status": "ok",
            })

        # Update cache after a full cycle
        LATEST_DATA["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LATEST_DATA["tickers"] = results
        LATEST_DATA["is_fetching"] = False
        print("Cycle complete! Cache updated.")

        # Wait before the next full cycle
        time.sleep(60)

# Start background thread (daemon so it stops when flask stops)
fetch_thread = threading.Thread(target=background_fetch_loop, daemon=True)
fetch_thread.start()


@app.route("/api/data")
def get_stock_data():
    """Return the cached data instantly."""
    if LATEST_DATA["is_fetching"]:
        # Return 202 Accepted to indicate it's still being prepared
        return jsonify({
            "status": "fetching", 
            "message": "Initial data fetch in progress... Please wait ~60 seconds."
        }), 202
        
    return jsonify(LATEST_DATA)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "tickers": TICKERS})


if __name__ == "__main__":
    # use debug=False to prevent Flask from spawning a second worker process and duplicating loops
    app.run(debug=False, port=5000)
