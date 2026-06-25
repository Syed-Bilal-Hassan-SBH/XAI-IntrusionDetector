"""
Trainer module for XAI-E-DiD
Handles training loop, backpropagation, and model saving
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path
import logging

from models.gan_lstm_ae import GANLSTMAutoencoder
from training.loss_functions import CombinedLoss


class Trainer:
    """Trainer for GAN-LSTM-Autoencoder model"""
    
    def __init__(
        self,
        model: GANLSTMAutoencoder,
        learning_rate: float = 0.001,
        device: str = None,
        save_dir: str = "models"
    ):
        self.model = model
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = CombinedLoss()
        
        # Training history
        self.history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'gan_loss': [],
            'focal_loss': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader):
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': [],
            'reconstruction_loss': [],
            'gan_loss': [],
            'focal_loss': []
        }
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Get batch data
            if len(batch) == 2:
                x, y = batch
            else:
                x = batch[0]
                y = None
            
            x = x.to(self.device)
            
            # Forward pass
            reconstructed, error, attention_weights, discriminator_output = self.model(x)
            
            # Generate fake data for discriminator
            with torch.no_grad():
                fake_data = torch.randn_like(x)
                _, _, _, discriminator_output_fake = self.model(fake_data)
            
            # Compute loss
            total_loss, loss_dict = self.loss_fn(
                reconstructed, x,
                discriminator_output, discriminator_output_fake,
                y
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Record losses
            for key in epoch_losses.keys():
                epoch_losses[key].append(loss_dict[key])
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'rec': f"{loss_dict['reconstruction_loss']:.4f}",
                'gan': f"{loss_dict['gan_loss']:.4f}"
            })
        
        # Compute epoch averages
        epoch_avg = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return epoch_avg
    
    def validate(self, val_loader: DataLoader):
        """Validate the model"""
        self.model.eval()
        
        val_losses = {
            'total_loss': [],
            'reconstruction_loss': [],
            'gan_loss': [],
            'focal_loss': []
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if len(batch) == 2:
                    x, y = batch
                else:
                    x = batch[0]
                    y = None
                
                x = x.to(self.device)
                
                # Forward pass
                reconstructed, error, attention_weights, discriminator_output = self.model(x)
                
                # Generate fake data
                fake_data = torch.randn_like(x)
                _, _, _, discriminator_output_fake = self.model(fake_data)
                
                # Compute loss
                total_loss, loss_dict = self.loss_fn(
                    reconstructed, x,
                    discriminator_output, discriminator_output_fake,
                    y
                )
                
                # Record losses
                for key in val_losses.keys():
                    val_losses[key].append(loss_dict[key])
        
        # Compute validation averages
        val_avg = {key: np.mean(values) for key, values in val_losses.items()}
        
        return val_avg
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save the best model
        """
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        
        if val_loader:
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Record training losses
            for key in self.history.keys():
                self.history[key].append(train_losses[key])
            
            self.logger.info(f"Train Loss: {train_losses['total_loss']:.4f}")
            self.logger.info(f"  Reconstruction: {train_losses['reconstruction_loss']:.4f}")
            self.logger.info(f"  GAN: {train_losses['gan_loss']:.4f}")
            self.logger.info(f"  Focal: {train_losses['focal_loss']:.4f}")
            
            # Validate
            if val_loader:
                val_losses = self.validate(val_loader)
                
                self.logger.info(f"Val Loss: {val_losses['total_loss']:.4f}")
                self.logger.info(f"  Reconstruction: {val_losses['reconstruction_loss']:.4f}")
                self.logger.info(f"  GAN: {val_losses['gan_loss']:.4f}")
                self.logger.info(f"  Focal: {val_losses['focal_loss']:.4f}")
                
                # Early stopping
                if val_losses['total_loss'] < best_val_loss:
                    best_val_loss = val_losses['total_loss']
                    patience_counter = 0
                    
                    if save_best:
                        self.save_model(epoch, val_losses['total_loss'])
                        self.logger.info("Model saved (best validation loss)")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                # Save based on training loss if no validation
                if train_losses['total_loss'] < best_val_loss:
                    best_val_loss = train_losses['total_loss']
                    if save_best:
                        self.save_model(epoch, train_losses['total_loss'])
        
        self.logger.info("Training completed")
        self.save_history()
    
    def save_model(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint_path = self.save_dir / "model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }, checkpoint_path)
    
    def save_history(self):
        """Save training history"""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to {history_path}")
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.logger.info(f"Model loaded from {checkpoint_path}")


if __name__ == "__main__":
    # Test trainer
    from ..models.gan_lstm_ae import GANLSTMAutoencoder
    
    # Create dummy data
    batch_size = 32
    seq_len = 20
    input_dim = 784
    hidden_dim = 128
    
    # Create model
    model = GANLSTMAutoencoder(input_dim, hidden_dim, seq_len)
    
    # Create dummy dataset
    X = torch.randn(100, seq_len, input_dim)
    y = torch.randint(0, 2, (100,)).float()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create trainer
    trainer = Trainer(model, learning_rate=0.001)
    
    # Train for 2 epochs
    trainer.train(loader, epochs=2)
