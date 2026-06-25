"""
Configuration management for XAI-E-DiD
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            return self._get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'data': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv',
                'batch_size': 32,
                'num_workers': 4
            },
            'model': {
                'input_dim': 784,
                'hidden_dims': [256, 128, 64],
                'output_dim': 2,
                'dropout': 0.3
            },
            'training': {
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'early_stopping_patience': 10
            },
            'paths': {
                'model_save_dir': 'models',
                'results_dir': 'results',
                'experiments_dir': 'experiments'
            }
        }
    
    def save(self, config_path: str = None):
        """Save configuration to YAML file"""
        path = Path(config_path) if config_path else self.config_path
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
