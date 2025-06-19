import numpy as np
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.X = np.random.randn(n_samples, 10)
        self.y = (self.X.sum(axis=1) > 0).astype(int)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]