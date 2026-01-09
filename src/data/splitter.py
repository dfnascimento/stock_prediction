import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockDataset(Dataset):
    """Dataset personalizado para dados de ações"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataSplitter:
    """Divisão de dados sem visualização"""
    
    @staticmethod
    def split_data(X, y, train_ratio, val_ratio):
        """Divide dados em treino, validação e teste"""
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        splits = {
            'train': (X[:train_end], y[:train_end]),
            'val': (X[train_end:val_end], y[train_end:val_end]),
            'test': (X[val_end:], y[val_end:])
        }
        
        # Estatísticas
        stats = {}
        for split_name, (X_split, y_split) in splits.items():
            stats[split_name] = {
                'samples': len(X_split),
                'percentage': len(X_split) / total_samples * 100
            }
        
        return splits, stats
    
    @staticmethod
    def create_dataloaders(splits, batch_size=32):
        """Cria DataLoaders para PyTorch"""
        dataloaders = {}
        
        for split_name, (X, y) in splits.items():
            dataset = StockDataset(X, y)
            shuffle = (split_name == 'train')
            dataloaders[split_name] = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle
            )
        
        return dataloaders