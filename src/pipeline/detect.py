"""
Detection pipeline for XAI-E-DiD
Takes input sample, preprocesses, passes through model, computes reconstruction error
"""

import torch
import numpy as np
import joblib
from pathlib import Path

from models.gan_lstm_ae import GANLSTMAutoencoder, AnomalyDetector
from data.preprocessing import apply_preprocessing
from data.vif_filter import apply_vif_filter
from data.feature_engineering import create_sequences


class DetectionPipeline:
    """Pipeline for anomaly detection"""
    
    def __init__(self, model_path: str = "models/model.pt", scaler_path: str = "models/scaler.pkl"):
        """
        Initialize detection pipeline.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Load feature info if exists
        feature_info_path = Path(model_path).parent / "feature_info.pkl"
        if feature_info_path.exists():
            self.feature_info = joblib.load(feature_info_path)
            self.remaining_features = self.feature_info['remaining_features']
        else:
            self.remaining_features = None
        
        # Load model metadata
        metadata_path = Path(model_path).parent / "model_metadata.pkl"
        if metadata_path.exists():
            self.metadata = joblib.load(metadata_path)
            self.seq_len = self.metadata['seq_len']
        else:
            self.seq_len = 20
        
        # Create anomaly detector
        self.detector = AnomalyDetector(self.model)
        self.threshold = None
    
    def _load_model(self, model_path: str) -> GANLSTMAutoencoder:
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model parameters from checkpoint or metadata
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Try to load metadata
        metadata_path = Path(model_path).parent / "model_metadata.pkl"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            model = GANLSTMAutoencoder(
                input_dim=metadata['input_dim'],
                hidden_dim=metadata['hidden_dim'],
                seq_len=metadata['seq_len']
            )
        else:
            # Try to infer from state dict
            # Default values
            model = GANLSTMAutoencoder(
                input_dim=784,
                hidden_dim=128,
                seq_len=20
            )
        
        model.load_state_dict(state_dict)
        return model
    
    def preprocess(self, sample: np.ndarray) -> np.ndarray:
        """
        Preprocess input sample.
        
        Args:
            sample: Input data (2D array or DataFrame)
        
        Returns:
            Preprocessed data
        """
        import pandas as pd
        
        # Convert to DataFrame if needed
        if not isinstance(sample, pd.DataFrame):
            sample = pd.DataFrame(sample)
        
        # Apply preprocessing
        sample_clean = apply_preprocessing(sample, self.scaler)
        
        # Apply VIF filter if features were removed
        if self.remaining_features:
            sample_clean = apply_vif_filter(sample_clean, self.remaining_features)
        
        return sample_clean.values
    
    def create_sequence(self, sample: np.ndarray) -> np.ndarray:
        """
        Create sequence from sample.
        
        Args:
            sample: Preprocessed sample data
        
        Returns:
            Sequence of shape (1, seq_len, n_features)
        """
        # Ensure we have enough data for sequence
        if len(sample) < self.seq_len:
            # Pad with zeros if not enough data
            padding = np.zeros((self.seq_len - len(sample), sample.shape[1]))
            sample = np.vstack([padding, sample])
        
        # Take last seq_len samples
        sequence = sample[-self.seq_len:]
        
        # Add batch dimension
        sequence = sequence.reshape(1, self.seq_len, -1)
        
        return sequence
    
    def compute_reconstruction_error(self, sample: np.ndarray) -> float:
        """
        Compute reconstruction error for sample.
        
        Args:
            sample: Preprocessed sequence
        
        Returns:
            Reconstruction error
        """
        with torch.no_grad():
            sample_tensor = torch.FloatTensor(sample).to(self.device)
            reconstructed, error, _, _ = self.model(sample_tensor)
            error_value = error.item()
        
        return error_value
    
    def detect_anomaly(self, sample: np.ndarray) -> dict:
        """
        Detect anomaly in input sample.
        
        Args:
            sample: Input data (2D array)
        
        Returns:
            Dictionary with detection results
        """
        # Preprocess
        sample_clean = self.preprocess(sample)
        
        # Create sequence
        sequence = self.create_sequence(sample_clean)
        
        # Compute reconstruction error
        error = self.compute_reconstruction_error(sequence)
        
        # Determine if anomaly
        if self.threshold is None:
            is_anomaly = None
            confidence = None
        else:
            is_anomaly = error > self.threshold
            confidence = (error - self.threshold) / self.threshold if is_anomaly else 1 - (error / self.threshold)
        
        result = {
            'reconstruction_error': error,
            'threshold': self.threshold,
            'is_anomaly': is_anomaly,
            'confidence': confidence
        }
        
        return result
    
    def set_threshold(self, threshold: float):
        """Set anomaly detection threshold"""
        self.threshold = threshold
        self.detector.set_threshold(threshold)
    
    def compute_threshold(self, normal_samples: np.ndarray, multiplier: float = 2.0) -> float:
        """
        Compute threshold from normal samples.
        
        Args:
            normal_samples: Array of normal samples
            multiplier: Multiplier for std (default: 2.0)
        
        Returns:
            Computed threshold (mean + 2 * std)
        """
        errors = []
        
        for sample in normal_samples:
            sample_clean = self.preprocess(sample)
            sequence = self.create_sequence(sample_clean)
            error = self.compute_reconstruction_error(sequence)
            errors.append(error)
        
        errors = np.array(errors)
        
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        threshold = mean_error + multiplier * std_error
        
        self.set_threshold(threshold)
        
        print(f"Computed threshold: {threshold:.6f}")
        print(f"Mean error: {mean_error:.6f}")
        print(f"Std error: {std_error:.6f}")
        
        return threshold


if __name__ == "__main__":
    # Test detection pipeline
    import sys
    sys.path.append("..")
    
    # Create dummy model for testing
    from ..models.gan_lstm_ae import GANLSTMAutoencoder
    import torch
    
    model = GANLSTMAutoencoder(784, 128, 20)
    torch.save(model.state_dict(), "models/model_test.pt")
    
    # Create dummy scaler
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    import pandas as pd
    
    scaler = MinMaxScaler()
    dummy_data = np.random.randn(100, 784)
    scaler.fit(dummy_data)
    joblib.dump(scaler, "models/scaler_test.pkl")
    
    # Test pipeline
    pipeline = DetectionPipeline("models/model_test.pt", "models/scaler_test.pkl")
    
    # Create dummy sample
    sample = np.random.randn(20, 784)
    result = pipeline.detect_anomaly(sample)
    
    print(f"Detection result: {result}")
