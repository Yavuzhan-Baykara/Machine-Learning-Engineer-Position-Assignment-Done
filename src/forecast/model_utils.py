import torch
import torch.nn as nn
import joblib
import io
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=4, hidden_size=128):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)  # Tek kolon (örneğin, Close tahmini)

    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        output, (hidden, cell) = self.lstm(x)
        x = self.fc(hidden)   # hidden: (1, batch, hidden_size)
        return x.squeeze(0)   # (batch, 1)

def load_model_from_buffer(buffer, input_size=4, hidden_size=128):
    model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_scaler_from_buffer(buffer):
    scaler = joblib.load(buffer)
    return scaler

def load_model_and_scaler(model_source, scaler_source, file_source="local"):
    """
    Loads the model and scaler from either local file paths or uploaded buffers.
    """
    if file_source == "uploaded":
        model_buffer = io.BytesIO(model_source.read())
        scaler_buffer = io.BytesIO(scaler_source.read())
        model = load_model_from_buffer(model_buffer)
        scaler = load_scaler_from_buffer(scaler_buffer)
    elif file_source == "local":
        model = NeuralNetwork(input_size=4, hidden_size=128)
        model.load_state_dict(torch.load(model_source, map_location=torch.device('cpu')))
        model.eval()
        scaler = joblib.load(scaler_source)
    else:
        raise ValueError("file_source must be either 'local' or 'uploaded'")
    return model, scaler

def normalize_sequence(sequence, scaler):
    """
    Normalizes the given sequence using the provided scaler.
    
    If scaler is fitted on 1 feature, each column is normalized individually;
    otherwise, the entire sequence is normalized.
    
    Args:
      sequence (np.array): shape (seq_length, num_features)
      scaler: loaded scaler
    
    Returns:
      np.array: normalized sequence with same shape.
    """
    if scaler.n_features_in_ == 1:
        sequence_scaled = np.zeros_like(sequence)
        for i in range(sequence.shape[1]):
            sequence_scaled[:, i] = scaler.transform(sequence[:, i].reshape(-1,1)).ravel()
    else:
        sequence_scaled = scaler.transform(sequence)
    return sequence_scaled
