import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 🧠 Load your pre-trained model
# -------------------------------
# Replace this with your actual model path
import pickle

with open('Project-Stock_Market/model1.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']

with open('Project-Stock_Market/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# scaler = data['scaler']

# Now you can use the scaler and model for predictions
with open('Project-Stock_Market/X_Test.pkl', 'rb') as f:
    X_scaled = pickle.load(f)
 
y_pred = model.predict(X_scaled)


# Initialize scaler (make sure it matches training setup)
scaler = MinMaxScaler(feature_range=(0, 1))

# ------------------------------------
# 🎨 Streamlit Page Configuration
# ------------------------------------
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📊 AI Stock Price Prediction Dashboard")

st.sidebar.header("⚙️ Options")

# ------------------------------------
# 🏦 User Inputs
# ------------------------------------
available_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", "ADBE"]

selected_stock = st.sidebar.selectbox("Select Stock", available_stocks)
compare_stock = st.sidebar.selectbox("Compare With (optional)", ["None"] + available_stocks)
days_to_predict = st.sidebar.slider("Days to Predict", 1, 30, 7)

st.sidebar.info("Model predicts closing prices using LSTM model trained on past data.")

# ------------------------------------
# 📥 Download Data
# ------------------------------------
@st.cache_data
def get_stock_data(ticker):
    end = datetime.now()
    start = end - timedelta(days=365 * 2)
    data = yf.download(ticker, start=start, end=end)
    return data

def predict_future_prices(data, n_days=7):
    # Use only 'Close' price
    close_prices = data[['Close']]
    scaled_data = scaler.fit_transform(close_prices)

    # Take last 60 values as input sequence
    last_sequence = scaled_data[-60:]
    predictions = []

    for _ in range(n_days):
        seq_input = np.expand_dims(last_sequence, axis=0)
        pred = model.predict(seq_input)
        predictions.append(pred[0, 0])
        # Append and slide window
        last_sequence = np.append(last_sequence[1:], [[pred[0, 0]]], axis=0)

    # Reverse scaling
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    # Prepare result DataFrame
    future_dates = pd.date_range(data.index[-1] + timedelta(days=1), periods=n_days)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Close": predicted_prices})
    return pred_df

# ------------------------------------
# 🔮 Main Prediction
# ------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader(f"📈 {selected_stock} Future Prediction")
    data = get_stock_data(selected_stock)
    pred_df = predict_future_prices(data, days_to_predict)

    # Plot prediction graph
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index[-60:], data['Close'].tail(60), label="Actual (Last 60 Days)", linewidth=2)
    ax.plot(pred_df['Date'], pred_df['Predicted Close'], label="Predicted", linestyle='dashed', color='red')
    ax.legend()
    ax.set_title(f"{selected_stock} Price Forecast ({days_to_predict} days)")
    st.pyplot(fig)
    st.dataframe(pred_df.style.highlight_max(color='lightgreen'))

# ------------------------------------
# ⚖️ Compare with another stock
# ------------------------------------
if compare_stock != "None":
    with col2:
        st.subheader(f"📊 Comparing {selected_stock} vs {compare_stock}")
        compare_data = get_stock_data(compare_stock)
        compare_pred = predict_future_prices(compare_data, days_to_predict)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(pred_df['Date'], pred_df['Predicted Close'], label=f"{selected_stock}", linewidth=2)
        ax2.plot(compare_pred['Date'], compare_pred['Predicted Close'], label=f"{compare_stock}", linestyle='dashed')
        ax2.legend()
        ax2.set_title(f"{selected_stock} vs {compare_stock} Prediction Comparison")
        st.pyplot(fig2)

# ------------------------------------
# 🧾 Additional Info
# ------------------------------------
st.markdown("---")
st.markdown("💡 **Tip:** Predictions are based on historical closing prices using an LSTM neural network. "
            "Market conditions, volatility, and external events can affect actual outcomes.")

