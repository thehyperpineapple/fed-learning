import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.reset_index(drop=True)  # Reset indices to avoid indexing issues
        try:
            self.y = y.reset_index(drop=True)  # Reset indices to avoid indexing issues
        except:
            self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        try:
            X_tensor = torch.tensor(self.X.iloc[idx].values, dtype=torch.float32)
            y_tensor = torch.tensor(self.y.iloc[idx], dtype=torch.long)
            return X_tensor, y_tensor
        except TypeError:
            self._check_indexing_error(idx)
        except Exception as e:
            print(f"Unexpected error: {e}, Index: {idx}")

    def _check_indexing_error(self, idx):
        if isinstance(idx, (list, tuple, pd.Index)):
            raise IndexError("Invalid index provided. Index should be an integer.")
        raise