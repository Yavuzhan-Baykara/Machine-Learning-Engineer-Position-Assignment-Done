import joblib
import torch
import numpy as np
import pandas as pd

def load_model_and_scaler(model_file, scaler_file):
    """
    Yüklenen model ve scaler'ı belleğe alır.
    
    Args:
    - model_file (uploaded file): .pt uzantılı model dosyası
    - scaler_file (uploaded file): .pkl uzantılı scaler dosyası
    
    Returns:
    - model, scaler: Yüklenmiş PyTorch modeli ve sklearn scaler nesnesi
    """
    model = torch.jit.load(model_file)
    scaler = joblib.load(scaler_file)
    return model, scaler

def normalize_sequence(sequence, scaler):
    """
    Verilen diziyi normalize eder.

    Args:
    - sequence (np.array): Giriş verisi
    - scaler (sklearn Scaler): MinMaxScaler
    
    Returns:
    - np.array: Normalize edilmiş dizi
    """
    return scaler.transform(sequence)

def add_technical_indicators(df):
    """
    Adds SMA (5, 9, 17), MACD, and other technical indicators to the DataFrame.
    
    Args:
    - df (pd.DataFrame): Stock data with 'Close' prices.
    
    Returns:
    - df (pd.DataFrame): DataFrame with added indicators.
    """
    # Simple Moving Averages (SMA)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_9'] = df['Close'].rolling(window=9).mean()
    df['SMA_17'] = df['Close'].rolling(window=17).mean()

    # NaN değerlerini dolduralım
    df[['SMA_5', 'SMA_9', 'SMA_17']] = df[['SMA_5', 'SMA_9', 'SMA_17']].fillna(method='bfill').fillna(method='ffill')

    # MACD Calculation
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA
    df['MACD'] = df['EMA_12'] - df['EMA_26']  # MACD line
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line

    # **Sütun isimlerini koruyalım**
    df = df.rename(columns={
        "SMA_5": "SMA (5 Days)",
        "SMA_9": "SMA (9 Days)",
        "SMA_17": "SMA (17 Days)",
        "MACD": "MACD",
        "MACD_Signal": "MACD Signal",
        "EMA_12": "EMA (12 Days)",
        "EMA_26": "EMA (26 Days)"
    })

    return df
