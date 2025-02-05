import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

# 🔹 Your Polygon.io API Key (Already Inserted)
POLYGON_API_KEY = "RFFDShqsKc_lGTkbTdZmsrWppKOO1R9S"

# 🔹 Fetch all U.S. stocks from Polygon.io
def get_stock_list():
    url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&limit=1000&apikey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=10)  # Timeout to prevent freezing
        response.raise_for_status()  # Raises an error for bad responses (4xx, 5xx)
        data = response.json()

        if "results" not in data:
            st.warning("⚠️ No stock data found. Check API key and usage limits.")
            return []

        stock_list = [stock["ticker"] for stock in data["results"]]
        return stock_list[:50]  # Limit scan to first 50 stocks (Adjust as needed)

    except requests.RequestException as e:
        st.error(f"🚨 API Request Error: {e}")
        return []

# 🔹 Fetch stock data from Polygon.io with error handling
def get_stock_data(symbol):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apikey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            st.warning(f"⚠️ No stock data found for {symbol}. Skipping...")
            return None

        stock_info = data["results"][0]
        return {
            "symbol": symbol,
            "close": stock_info.get("c", None),  # Validate key existence
            "volume": stock_info.get("v", None),
            "high": stock_info.get("h", None),
            "low": stock_info.get("l", None)
        }

    except requests.RequestException as e:
        st.error(f"🚨 Failed to fetch data for {symbol}: {e}")
        return None

# 🔹 Fetch news sentiment score with error handling
def get_news_sentiment(symbol):
    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&apikey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            st.warning(f"⚠️ No news found for {symbol}. Sentiment set to neutral (0).")
            return 0  # Default neutral sentiment if no news found

        sentiment_score = 0
        count = 0
        for article in data["results"][:5]:  # Use latest 5 articles
            if "bullish" in article["title"].lower():
                sentiment_score += 1
            elif "bearish" in article["title"].lower():
                sentiment_score -= 1
            count += 1

        return sentiment_score / count if count > 0 else 0

    except requests.RequestException as e:
        st.error(f"🚨 Failed to fetch news sentiment for {symbol}: {e}")
        return 0  # Default neutral sentiment in case of failure

# 🔹 AI Model for Breakout Prediction with Data Validation
def predict_breakout(stock_data, sentiment_score):
    try:
        # Validate stock data
        if stock_data["close"] is None or stock_data["volume"] is None:
            st.warning(f"⚠️ Incomplete data for {stock_data['symbol']}. Skipping...")
            return None

        # Sample training data (Replace with real historical data)
        training_data = pd.DataFrame({
            "close": [150, 152, 148, 151, 155, 160, 162, 158],
            "volume": [1_000_000, 1_200_000, 900_000, 1_100_000, 1_300_000, 1_500_000, 1_600_000, 1_400_000],
            "sentiment": [0.6, 0.8, 0.4, 0.7, 0.9, 1.0, 1.0, 0.8],
            "breakout": [0, 1, 0, 1, 1, 1, 1, 0]
        })

        X = training_data[["close", "volume", "sentiment"]]
        y = training_data["breakout"]
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)

        # ✅ FIX: Convert input to Pandas DataFrame with column names
        new_data = pd.DataFrame([[stock_data["close"], stock_data["volume"], sentiment_score]],
                                columns=["close", "volume", "sentiment"])

        prediction = model.predict_proba(new_data)
        return prediction[0][1] * 100  # Probability of breakout

    except Exception as e:
        st.error(f"🚨 AI Model Error: {e}")
        return None

# 🔹 Web App UI (Streamlit)
st.title("📈 AI Stock Scanner for Breakouts")
st.write("Scan the entire U.S. stock market for stocks with high breakout probability.")

# 🔹 Scan Button with Progress Bar
if st.button("Start Scan"):
    stock_list = get_stock_list()
    if not stock_list:
        st.error("🚨 No stocks found. Check API connection and usage limits.")
    else:
        total_stocks = len(stock_list)

        # Progress bar setup
        progress_bar = st.progress(0)
        status_text = st.empty()

        breakout_candidates = []
        estimated_time_per_stock = 0.5  # Adjust based on API speed
        total_estimated_time = total_stocks * estimated_time_per_stock

        st.write(f"🔄 Estimated scan time: **{total_estimated_time:.1f} seconds** ⏳")

        for i, stock in enumerate(stock_list):
            stock_data = get_stock_data(stock)
            sentiment_score = get_news_sentiment(stock)

            if stock_data:
                breakout_prob = predict_breakout(stock_data, sentiment_score)
                if breakout_prob and breakout_prob > 75:  # Show stocks with 75%+ breakout probability
                    breakout_candidates.append((stock, breakout_prob))

            # Update progress bar
            progress_bar.progress((i + 1) / total_stocks)
            status_text.text(f"Scanning {i + 1} of {total_stocks} stocks...")

            # Simulating estimated time per scan
            time.sleep(estimated_time_per_stock)

        # Sort results
        breakout_candidates.sort(key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(breakout_candidates, columns=["Stock", "Breakout Probability (%)"])

        # Final Output
        progress_bar.progress(1.0)
        status_text.text("✅ Scan Complete!")

        if not df.empty:
            st.write("### 📊 Top Breakout Stocks:")
            st.dataframe(df)
        else:
            st.write("No high-probability setups found.")

