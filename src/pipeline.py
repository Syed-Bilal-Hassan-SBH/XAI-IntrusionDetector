"""
Main pipeline for XAI-E-DiD
"""

from pathlib import Path
import torch

from .config import Config
from .data_loader import DataLoader
from .model import ModelFactory
from .trainer import Trainer
from .evaluator import Evaluator


class Pipeline:
    """Main pipeline for training and evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.model = None
        self.trainer = None
    
    def train(self):
        """Run training pipeline"""
        print("Initializing training pipeline...")
        
        # Create dataloaders
        train_path = self.config.get('data.train_path', 'data/train.csv')
        train_loader, val_loader = self.data_loader.create_dataloaders(train_path)
        
        # Create model
        self.model = ModelFactory.create_model(self.config)
        print(f"Model created: {self.model}")
        
        # Initialize trainer
        self.trainer = Trainer(self.model, self.config)
        
        # Train
        history = self.trainer.train(train_loader, val_loader)
        self.trainer.save_history()
        
        return history
    
    def evaluate(self, model_path: str):
        """Run evaluation pipeline"""
        print("Initializing evaluation pipeline...")
        
        # Create model
        self.model = ModelFactory.create_model(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        
        # Create dataloaders
        test_path = self.config.get('data.test_path', 'data/test.csv')
        train_path = self.config.get('data.train_path', 'data/train.csv')
        _, test_loader = self.data_loader.create_dataloaders(train_path, test_path)
        
        # Evaluate
        evaluator = Evaluator(self.model, self.config)
        results = evaluator.evaluate(test_loader)
        
        return results
