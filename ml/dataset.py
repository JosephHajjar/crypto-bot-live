import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from data.feature_engineer_btc import get_feature_cols

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for Time-Series forecasting.
    Dynamically reads feature columns from the canonical list.
    """
    def __init__(self, data_path, seq_length=60, feature_cols=None):
        print(f"Loading engineered dataset from {data_path}...")
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.seq_length = seq_length
        
        self.feature_cols = feature_cols if feature_cols is not None else get_feature_cols()
        
        # Filter to only columns that exist in the dataframe
        available = [c for c in self.feature_cols if c in self.df.columns]
        if len(available) < len(self.feature_cols):
            missing = set(self.feature_cols) - set(available)
            print(f"  Warning: {len(missing)} features missing: {missing}")
        self.feature_cols = available
        
        # Extract numpy arrays for speed
        self.X = self.df[self.feature_cols].values.astype(np.float32)
        self.y = self.df['Target'].values.astype(np.int64)
        
        # Replace any remaining NaN/inf with 0
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.length = len(self.X) - self.seq_length
        print(f"Dataset: {self.length} sequences of length {seq_length}, {len(self.feature_cols)} features.")
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_length]
        y_label = self.y[idx + self.seq_length - 1]
        
        x_tensor = torch.from_numpy(x_seq)
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        
        return x_tensor, y_tensor

if __name__ == "__main__":
    import os
    path = "data_storage/BTC_USDT_15m_processed.csv"
    if os.path.exists(path):
        ds = TimeSeriesDataset(path, seq_length=60)
        x, y = ds[0]
        print(f"X shape: {x.shape}, Y: {y}")
    else:
        print(f"Run feature engineering first.")
