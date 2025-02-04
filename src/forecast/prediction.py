import numpy as np
import torch
from src.forecast.model_utils import normalize_sequence

def predict_next_days(open_val, high_val, low_val, sma_val, model, scaler, forecast_days):
    """
    Forecasts the stock price for a given number of days using the model.
    
    Args:
      open_val, high_val, low_val, sma_val (float): Input features.
      model: Loaded PyTorch model.
      scaler: Loaded MinMaxScaler.
      forecast_days (int): Number of days to forecast.
      
    Returns:
      dates (list): List of datetime objects for the forecast period.
      y_pred_denorm (np.array): Denormalized predicted close prices.
    """
    seq_length = 11  # Fixed sequence length
    input_row = np.array([open_val, high_val, low_val, sma_val], dtype=np.float32)
    sequence = np.tile(input_row, (seq_length, 1))  # (seq_length, 4)
    
    predictions = []
    for _ in range(forecast_days):
        seq_scaled = normalize_sequence(sequence, scaler)
        x_input = torch.tensor(seq_scaled, dtype=torch.float).unsqueeze(0)  # (1, seq_length, 4)
        with torch.no_grad():
            y_pred = model(x_input)
        y_pred_value = y_pred.squeeze().item()
        dummy = np.zeros((1, scaler.n_features_in_), dtype=np.float32)
        # Assume the "Close" prediction corresponds to column index 3
        dummy[0, 3] = y_pred_value
        y_pred_denorm_value = scaler.inverse_transform(dummy)[0, 3]
        predictions.append(y_pred_denorm_value)
        new_input = np.array([open_val, high_val, low_val, sma_val], dtype=np.float32)
        sequence = np.vstack([sequence[1:], new_input])
    
    from datetime import datetime, timedelta
    dates = [datetime.today() + timedelta(days=i) for i in range(forecast_days)]
    return dates, np.array(predictions)
