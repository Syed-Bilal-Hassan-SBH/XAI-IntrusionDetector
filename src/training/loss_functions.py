"""
Loss functions for XAI-E-DiD
Implements MSE, BCE, Focal loss and combined loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """MSE loss for reconstruction"""
    
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, reconstructed, original):
        """
        Compute MSE reconstruction loss.
        
        Args:
            reconstructed: Reconstructed sequence
            original: Original sequence
        
        Returns:
            loss: MSE loss
        """
        return self.mse(reconstructed, original)


class GANLoss(nn.Module):
    """Binary Cross Entropy loss for GAN discriminator"""
    
    def __init__(self):
        super(GANLoss, self).__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, discriminator_output, target):
        """
        Compute BCE loss for discriminator.
        
        Args:
            discriminator_output: Discriminator output (probability)
            target: Target labels (0 for fake, 1 for real)
        
        Returns:
            loss: BCE loss
        """
        return self.bce(discriminator_output, target)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        """
        Compute focal loss.
        
        Args:
            predictions: Predicted probabilities
            targets: Target labels
        
        Returns:
            loss: Focal loss
        """
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss for the GAN-LSTM-Autoencoder model"""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        gan_weight: float = 0.5,
        focal_weight: float = 0.1,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super(CombinedLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.gan_weight = gan_weight
        self.focal_weight = focal_weight
        
        self.reconstruction_loss = ReconstructionLoss()
        self.gan_loss = GANLoss()
        self.focal_loss = FocalLoss(alpha, gamma)
    
    def forward(
        self,
        reconstructed,
        original,
        discriminator_output_real,
        discriminator_output_fake,
        targets
    ):
        """
        Compute combined loss.
        
        Args:
            reconstructed: Reconstructed sequence
            original: Original sequence
            discriminator_output_real: Discriminator output on real data
            discriminator_output_fake: Discriminator output on fake data
            targets: Target labels for focal loss
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss
        rec_loss = self.reconstruction_loss(reconstructed, original)
        
        # GAN losses
        # Discriminator loss: real should be classified as 1, fake as 0
        real_target = torch.ones_like(discriminator_output_real)
        fake_target = torch.zeros_like(discriminator_output_fake)
        
        gan_loss_real = self.gan_loss(discriminator_output_real, real_target)
        gan_loss_fake = self.gan_loss(discriminator_output_fake, fake_target)
        gan_loss = gan_loss_real + gan_loss_fake
        
        # Focal loss (for classification if needed)
        if targets is not None:
            focal_loss = self.focal_loss(discriminator_output_real, targets)
        else:
            focal_loss = torch.tensor(0.0, device=reconstructed.device)
        
        # Combined loss
        total_loss = (
            self.reconstruction_weight * rec_loss +
            self.gan_weight * gan_loss +
            self.focal_weight * focal_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'reconstruction_loss': rec_loss.item(),
            'gan_loss': gan_loss.item(),
            'focal_loss': focal_loss.item()
        }
        
        return total_loss, loss_dict


class AdversarialLoss(nn.Module):
    """Adversarial loss for training the generator"""
    
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, discriminator_output_fake):
        """
        Compute adversarial loss for generator.
        Generator wants discriminator to classify fake as real.
        
        Args:
            discriminator_output_fake: Discriminator output on fake data
        
        Returns:
            loss: Adversarial loss
        """
        target = torch.ones_like(discriminator_output_fake)
        return self.bce(discriminator_output_fake, target)


if __name__ == "__main__":
    # Test loss functions
    batch_size = 32
    seq_len = 20
    input_dim = 784
    
    # Create dummy data
    original = torch.randn(batch_size, seq_len, input_dim)
    reconstructed = torch.randn(batch_size, seq_len, input_dim)
    discriminator_output_real = torch.rand(batch_size, 1)
    discriminator_output_fake = torch.rand(batch_size, 1)
    targets = torch.randint(0, 2, (batch_size, 1)).float()
    
    # Test reconstruction loss
    rec_loss = ReconstructionLoss()
    loss = rec_loss(reconstructed, original)
    print(f"Reconstruction loss: {loss.item():.4f}")
    
    # Test GAN loss
    gan_loss = GANLoss()
    loss = gan_loss(discriminator_output_real, targets)
    print(f"GAN loss: {loss.item():.4f}")
    
    # Test focal loss
    focal_loss = FocalLoss()
    loss = focal_loss(discriminator_output_real, targets)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss()
    total_loss, loss_dict = combined_loss(
        reconstructed, original,
        discriminator_output_real, discriminator_output_fake,
        targets
    )
    print(f"\nCombined loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
