import os
import streamlit as st
import pandas as pd
from src.preprocessing import add_technical_indicators

def get_local_file_pairs(model_dir, scaler_dir):
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")] if os.path.exists(model_dir) else []
    scaler_files = [f for f in os.listdir(scaler_dir) if f.endswith(".pkl")] if os.path.exists(scaler_dir) else []
    model_basenames = {os.path.splitext(f)[0] for f in model_files}
    scaler_basenames = {os.path.splitext(f)[0] for f in scaler_files}
    return sorted(list(model_basenames.intersection(scaler_basenames)))

def get_local_paths(model_dir, scaler_dir, base_name):
    return (
        os.path.join(model_dir, base_name + ".pt"),
        os.path.join(scaler_dir, base_name + ".pkl"),
    )

def validate_model_scaler(model_path, scaler_path):
    return os.path.exists(model_path) and os.path.exists(scaler_path)

def csv_download_button(df, filename):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )

def process_stock_data(df):
    """
    Process stock data by adding SMA and technical indicators.
    """
    df = add_technical_indicators(df)
    return df
