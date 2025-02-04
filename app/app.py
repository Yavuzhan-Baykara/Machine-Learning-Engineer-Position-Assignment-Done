import streamlit as st
import pandas as pd
import os
import sys
import io
import torch
import numpy as np

# Üst dizini sys.path'e ekleyelim
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    get_local_file_pairs, get_local_paths, validate_model_scaler, 
    csv_download_button, process_stock_data
)
from src.yahoofinance import get_last_30_days_data
from src.forecast.prediction import predict_next_days
from src.forecast.model_utils import load_model_and_scaler
from src.visualization import display_forecast_table, display_forecast_plot
from src.forecast.model_utils import normalize_sequence

# Streamlit genişlik ayarı
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 90%;
        padding-top: 20px;
        padding-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Ayarları ---
st.sidebar.header("Settings")
default_tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
selected_ticker = st.sidebar.selectbox("Select Ticker", options=default_tickers)
forecast_days = st.sidebar.slider("Number of days to forecast", min_value=1, max_value=50, value=30)

# Model ve Scaler Dosya Yolları (yerel)
model_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "model")
scaler_dir = os.path.join(os.path.dirname(__file__), "..", "weights", "Scalers")
local_files = get_local_file_pairs(model_dir, scaler_dir)

if local_files:
    selected_model = st.sidebar.selectbox("Select Model", options=local_files)
else:
    selected_model = None
    st.sidebar.warning("⚠️ No valid local model and scaler pairs found. Please upload files.")

# Model ve Scaler dosyalarını yükleme alanı (Upload)
uploaded_model = st.sidebar.file_uploader("Upload Model Weights (.pt)", type=["pt"])
uploaded_scaler = st.sidebar.file_uploader("Upload Scaler (.pkl)", type=["pkl"])

# Dosya kaynağı belirleme
file_source = "none"
model_path, scaler_path = None, None
if uploaded_model and uploaded_scaler:
    file_source = "uploaded"
    st.sidebar.success("✅ Uploaded model and scaler detected.")
elif selected_model:
    model_path, scaler_path = get_local_paths(model_dir, scaler_dir, selected_model)
    if validate_model_scaler(model_path, scaler_path):
        file_source = "local"
        st.sidebar.success(f"✅ Using local model: {selected_model}")
    else:
        st.sidebar.error("⚠️ Model or scaler file is missing in local directory.")
else:
    st.sidebar.warning("⚠️ No valid model or scaler file found. Please upload both files.")

# --- Main Content ---
st.title("Stock Analysis and Forecast App 📊")
tabs = st.tabs(["📈 Yahoo Finance Data", "🔮 Forecast"])

# ---------- Tab 1: Yahoo Finance Data ----------
with tabs[0]:
    st.header("Yahoo Finance Data Viewer")
    st.write("View the last 30 days of daily data for the selected ticker.")
    try:
        data = get_last_30_days_data(selected_ticker)
        if data.empty:
            st.warning("⚠️ No data was found for the selected ticker.")
        else:
            data = process_stock_data(data)  # SMA ve teknik verileri ekle
            st.success(f"📊 Data for **{selected_ticker}** from the last 30 days:")

            # **Sütun isimlerini zorunlu göster**
            data.columns = [str(col) for col in data.columns]

            st.dataframe(data, use_container_width=True)
            csv_download_button(data, f"{selected_ticker}_last30days.csv")
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}")


# ---------- Tab 2: Forecast ----------
with tabs[1]:
    st.header("Stock Price Forecast")
    st.write("Use the selected model to forecast future stock prices.")
    if file_source == "none":
        st.warning("⚠️ Please upload both the model (.pt) and scaler (.pkl) files to perform forecasting.")
    else:
        if file_source == "uploaded":
            model, scaler = load_model_and_scaler(uploaded_model, uploaded_scaler, file_source="uploaded")
        else:  # file_source == "local"
            model, scaler = load_model_and_scaler(model_path, scaler_path, file_source="local")

        if st.sidebar.button(f"Run {forecast_days}-Day Forecast"):
            # SMA verilerini al
            data = get_last_30_days_data(selected_ticker)
            data = process_stock_data(data)

            # En son satırdaki SMA verilerini al
            latest_data = data.iloc[-1]

            # **Doğru formatta float değerler al**
            open_val = float(latest_data['Open'])
            high_val = float(latest_data['High'])
            low_val = float(latest_data['Low'])
            sma_val = float(latest_data['SMA (17 Days)'])  # Model için en iyi SMA değeri

            # Model ile tahmin yap
            dates, y_pred_denorm = predict_next_days(open_val, high_val, low_val, sma_val, model, scaler, forecast_days)

            # **Sonuçları Görselleştir**
            st.header(f"📈 {forecast_days}-Day Stock Price Prediction for {selected_ticker}")
            df_forecast = display_forecast_table(dates, y_pred_denorm)
            display_forecast_plot(dates, y_pred_denorm, selected_ticker, forecast_days)
            csv_download_button(df_forecast, f"{selected_ticker}_{forecast_days}day_forecast.csv")
