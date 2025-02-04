import numpy as np
import torch
from datetime import datetime, timedelta
from src.forecast.model_utils import normalize_sequence

def predict_next_days(open_val, high_val, low_val, sma_val, model, scaler, forecast_days):
    seq_length = 11
    input_row = np.array([open_val, high_val, low_val, sma_val], dtype=np.float32)
    sequence = np.tile(input_row, (seq_length, 1))  # (seq_length, 4)

    predictions = []
    
    for day in range(forecast_days):
        seq_scaled = normalize_sequence(sequence, scaler)
        x_input = torch.tensor(seq_scaled, dtype=torch.float).unsqueeze(0)  # (1, seq_length, 4)
        
        with torch.no_grad():
            y_pred = model(x_input)  # (1, 1)
        y_pred_value = y_pred.squeeze().item()

        num_features = scaler.n_features_in_
        if num_features == 1:
            y_pred_denorm_value = scaler.inverse_transform(np.array([[y_pred_value]]))[0, 0]
        else:
            dummy = np.zeros((1, num_features), dtype=np.float32)
            dummy[0, -1] = y_pred_value
            y_pred_denorm_value = scaler.inverse_transform(dummy)[0, -1]
        
        predictions.append(y_pred_denorm_value)
        
        new_close = y_pred_denorm_value
        new_open  = new_close               
        new_high  = new_close * 1.100     
        new_low   = new_close * 0.900  
        prev_sma = sequence[-1, 3]
        new_sma = (prev_sma * (seq_length - 1) + new_close) / seq_length
        
        new_input = np.array([new_open, new_high, new_low, new_sma], dtype=np.float32)
        sequence = np.vstack([sequence[1:], new_input])
    
    dates = [datetime.today() + timedelta(days=i) for i in range(forecast_days)]
    
    return dates, np.array(predictions)
