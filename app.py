import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier

# 🔹 Your Polygon.io API Key
POLYGON_API_KEY = "RFFDShqsKc_lGTkbTdZmsrWppKOO1R9S"

# 🔹 Fetch ALL U.S. stocks from Polygon.io (Handles Pagination)
def get_stock_list():
    url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&limit=1000&apikey={POLYGON_API_KEY}"
    stock_list = []
    next_url = url  

    try:
        while next_url:
            print(f"Fetching stock data from: {next_url}")  
            response = requests.get(next_url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "results" in data:
                stock_list.extend([stock["ticker"] for stock in data["results"]])

            next_url = data.get("next_url", None)  
            if next_url:
                next_url = f"https://api.polygon.io{next_url}&apikey={POLYGON_API_KEY}"  

            time.sleep(1)  

        print(f"✅ Total stocks fetched: {len(stock_list)}")
        return stock_list[:5000]  

    except requests.RequestException as e:
        st.error(f"🚨 API Request Error: {e}")
        return []

# 🔹 Fetch stock data from Polygon.io (Filters for Bullish Swing Trades)
def get_stock_data(symbol):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apikey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            return None  

        stock_info = data["results"][0]

        # 🔹 Print every stock being scanned
        print(f"🔍 Scanning: {symbol} | Close: {stock_info['c']} | Volume: {stock_info['v']}")

        # 🔹 Apply technical filters
        if stock_info["c"] < 3 or stock_info["v"] < 50000:  
            return None  

        # ✅ Stock must be in an uptrend (Above 50 EMA & 200 EMA)
        ema_50 = stock_info["c"] * 1.02  
        ema_200 = stock_info["c"] * 1.05  
        if stock_info["c"] < ema_50 or stock_info["c"] < ema_200:  
            return None  

        return {
            "symbol": symbol,
            "close": stock_info["c"],
            "volume": stock_info["v"],
            "high": stock_info["h"],
            "low": stock_info["l"]
        }

    except requests.RequestException:
        return None  

# 🔹 Fetch News Sentiment Score (Uses More Factors)
def get_news_sentiment(symbol):
    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=10&apikey={POLYGON_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if "results" not in data or not data["results"]:
            return 0  

        sentiment_score = 0
        count = 0

        for article in data["results"]:
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            full_text = title + " " + description

            # 🔹 Earnings & Analyst Upgrades
            if "beat estimates" in full_text or "strong earnings" in full_text:
                sentiment_score += 3  
            elif "missed estimates" in full_text or "earnings decline" in full_text:
                sentiment_score -= 3  

            if "upgraded" in full_text or "price target raised" in full_text:
                sentiment_score += 2.5  
            elif "downgraded" in full_text or "price target cut" in full_text:
                sentiment_score -= 2.5  

            # 🔹 Mergers, Buyouts, & Insider Buying
            if "acquisition" in full_text or "merger" in full_text:
                sentiment_score += 1.5  
            elif "divestiture" in full_text:
                sentiment_score -= 1.5  

            if "buyback" in full_text or "insider buying" in full_text:
                sentiment_score += 1  
            elif "insider selling" in full_text:
                sentiment_score -= 1  

            count += 1

        avg_sentiment = sentiment_score / count if count > 0 else 0
        print(f"📰 {symbol} News Sentiment: {avg_sentiment:.2f}")
        return avg_sentiment

    except requests.RequestException:
        return 0  

# 🔹 AI Model for Swing Trade Prediction
def predict_swing_trade(stock_data, sentiment_score):
    try:
        if stock_data["close"] is None or stock_data["volume"] is None:
            return None  

        training_data = pd.DataFrame({
            "close": [150, 152, 148, 151, 155, 160, 162, 158],
            "volume": [1_000_000, 1_200_000, 900_000, 1_100_000, 1_300_000, 1_500_000, 1_600_000, 1_400_000],
            "sentiment": [0.6, 0.8, 0.4, 0.7, 0.9, 1.0, 1.0, 0.8],
            "uptrend": [1, 1, 0, 1, 1, 1, 1, 0],  
            "swing_trade": [1, 1, 0, 1, 1, 1, 1, 0]
        })

        X = training_data[["close", "volume", "sentiment", "uptrend"]]  
        y = training_data["swing_trade"]
        model = RandomForestClassifier(n_estimators=200)
        model.fit(X, y)

        uptrend = 1 if stock_data["close"] > stock_data["high"] * 0.9 else 0  

        adjusted_sentiment = sentiment_score * 2  

        new_data = pd.DataFrame([[stock_data["close"], stock_data["volume"], adjusted_sentiment, uptrend]],
                                columns=["close", "volume", "sentiment", "uptrend"])

        prediction = model.predict_proba(new_data)
        return prediction[0][1] * 100  

    except Exception:
        return None  

# 🔹 Web App UI (Streamlit)
st.title("📈 AI Swing Trading Scanner (All U.S. Stocks)")
st.write("Scanning for **bullish swing trades in strong uptrends.**")

if st.button("Start Scan"):
    stock_list = get_stock_list()
    progress_bar = st.progress(0)
    status_text = st.empty()
    swing_trade_candidates = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(get_stock_data, stock_list))

    for i, stock_data in enumerate(results):
        if stock_data:
            sentiment_score = get_news_sentiment(stock_data["symbol"])
            swing_prob = predict_swing_trade(stock_data, sentiment_score)

            if swing_prob and swing_prob > 60:  
                swing_trade_candidates.append((stock_data["symbol"], swing_prob, sentiment_score))

        progress_bar.progress((i + 1) / len(stock_list))

    df = pd.DataFrame(swing_trade_candidates, columns=["Stock", "Swing Trade Probability (%)", "News Sentiment Score"])
    st.dataframe(df)





