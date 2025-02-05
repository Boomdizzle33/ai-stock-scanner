import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 🔹 Your Polygon.io API Key (Already Inserted)
POLYGON_API_KEY = "RFFDShqsKc_lGTkbTdZmsrWppKOO1R9S"

# 🔹 Fetch all U.S. stocks from Polygon.io
def get_stock_list():
    url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&limit=1000&apikey={POLYGON_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    stock_list = [stock["ticker"] for stock in data["results"]]
    return stock_list[:50]  # Limit scan to first 50 stocks (Adjust for more)

# 🔹 Fetch stock data from Polygon.io
def get_stock_data(symbol):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apikey={POLYGON_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if "results" in data and data["results"]:
        stock_info = data["results"][0]
        return {
            "symbol": symbol,
            "close": stock_info["c"],
            "volume": stock_info["v"],
            "high": stock_info["h"],
            "low": stock_info["l"]
        }
    else:
        return None

# 🔹 Fetch news sentiment score
def get_news_sentiment(symbol):
    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&apikey={POLYGON_API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    sentiment_score = 0
    count = 0
    for article in data.get("results", [])[:5]:  # Get sentiment from latest 5 articles
        if "bullish" in article["title"].lower():
            sentiment_score += 1
        elif "bearish" in article["title"].lower():
            sentiment_score -= 1
        count += 1
    
    return sentiment_score / count if count > 0 else 0

# 🔹 AI Model for Breakout Prediction
def predict_breakout(stock_data, sentiment_score):
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

    # Predict Breakout Probability
    new_data = np.array([[stock_data["close"], stock_data["volume"], sentiment_score]])
    prediction = model.predict_proba(new_data)
    
    return prediction[0][1] * 100  # Probability of breakout

# 🔹 Web App UI (Streamlit)
st.title("📈 AI Stock Scanner for Breakouts")
st.write("Scan the entire U.S. stock market for stocks with high breakout probability.")

# 🔹 Scan Button
if st.button("Start Scan"):
    stock_list = get_stock_list()
    breakout_candidates = []

    for stock in stock_list:
        stock_data = get_stock_data(stock)
        sentiment_score = get_news_sentiment(stock)

        if stock_data:
            breakout_prob = predict_breakout(stock_data, sentiment_score)
            if breakout_prob > 75:  # Show stocks with 75%+ breakout probability
                breakout_candidates.append((stock, breakout_prob))

    # Sort results
    breakout_candidates.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(breakout_candidates, columns=["Stock", "Breakout Probability (%)"])

    if not df.empty:
        st.write("### 📊 Top Breakout Stocks:")
        st.dataframe(df)
    else:
        st.write("No high-probability setups found.")
