"""
PyTorch model definition for intrusion detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntrusionDetectionModel(nn.Module):
    """Neural network for intrusion detection"""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.3):
        super(IntrusionDetectionModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def predict_proba(self, x):
        """Return class probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


class ModelFactory:
    """Factory for creating models"""
    
    @staticmethod
    def create_model(config) -> IntrusionDetectionModel:
        """Create model from configuration"""
        input_dim = config.get('model.input_dim', 784)
        hidden_dims = config.get('model.hidden_dims', [256, 128, 64])
        output_dim = config.get('model.output_dim', 2)
        dropout = config.get('model.dropout', 0.3)
        
        return IntrusionDetectionModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout
        )
