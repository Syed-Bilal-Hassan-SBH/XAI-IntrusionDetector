"""
Data loading and preprocessing for XAI-E-DiD
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple


class IntrusionDataset(Dataset):
    """Dataset for intrusion detection data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DataLoader:
    """Data loader and preprocessor"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from CSV file"""
        if not Path(data_path).exists():
            # Generate synthetic data for demonstration
            print(f"Data file not found at {data_path}. Generating synthetic data...")
            return self._generate_synthetic_data()
        
        df = pd.read_csv(data_path)
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        return features, labels
    
    def _generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 784) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic intrusion detection data"""
        np.random.seed(42)
        features = np.random.randn(n_samples, n_features)
        # Create binary labels with class imbalance (typical for intrusion detection)
        labels = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        return features, labels
    
    def preprocess_data(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Preprocess features"""
        if fit:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        return features
    
    def create_dataloaders(
        self,
        train_path: str,
        test_path: str = None,
        test_size: float = 0.2
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create train and test dataloaders"""
        
        # Load training data
        train_features, train_labels = self.load_data(train_path)
        train_features = self.preprocess_data(train_features, fit=True)
        
        # Split train/val
        if test_path is None:
            train_features, val_features, train_labels, val_labels = train_test_split(
                train_features, train_labels, test_size=test_size, random_state=42, stratify=train_labels
            )
        else:
            val_features, val_labels = self.load_data(test_path)
            val_features = self.preprocess_data(val_features, fit=False)
        
        # Create datasets
        train_dataset = IntrusionDataset(train_features, train_labels)
        val_dataset = IntrusionDataset(val_features, val_labels)
        
        # Create dataloaders
        batch_size = self.config.get('data.batch_size', 32)
        num_workers = self.config.get('data.num_workers', 4)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, val_loader
