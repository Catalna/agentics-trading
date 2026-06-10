import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SequenceDataset(Dataset):
    """
    Transforms 2D tabular DataFrames into 3D sequences for PyTorch models.
    """
    def __init__(self, X: pd.DataFrame, y: pd.Series, seq_len: int = 60, scaler: StandardScaler = None, is_train: bool = True):
        # We need to scale data for neural networks
        self.seq_len = seq_len
        self.features_np = X.values
        self.labels_np = y.values
        
        if is_train:
            self.scaler = StandardScaler()
            self.features_np = self.scaler.fit_transform(self.features_np)
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided for validation/test data")
            self.scaler = scaler
            self.features_np = self.scaler.transform(self.features_np)
            
    def __len__(self):
        # We lose seq_len - 1 samples at the beginning
        return len(self.features_np) - self.seq_len + 1

    def __getitem__(self, idx):
        # LSTM expects (seq_len, features)
        # We'll return (seq_len, features) and let the training loop reshape for CNN if needed
        # (CNN expects features, seq_len)
        
        end_idx = idx + self.seq_len
        seq_x = self.features_np[idx:end_idx]
        seq_y = self.labels_np[end_idx - 1] # Label belongs to the last candle of the sequence
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.long)
