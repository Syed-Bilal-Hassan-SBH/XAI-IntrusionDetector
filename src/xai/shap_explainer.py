"""
SHAP explainer for XAI-E-DiD
Uses SHAP DeepExplainer with fallback to KernelExplainer
"""

import torch
import numpy as np
import joblib
from pathlib import Path
import shap


class SHAPExplainer:
    """SHAP explainer for model interpretation"""
    
    def __init__(self, model_path: str = "models/model.pt", scaler_path: str = "models/scaler.pkl"):
        """
        Initialize SHAP explainer.
        
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
        
        # Load metadata
        metadata_path = Path(model_path).parent / "model_metadata.pkl"
        if metadata_path.exists():
            self.metadata = joblib.load(metadata_path)
            self.seq_len = self.metadata['seq_len']
        else:
            self.seq_len = 20
        
        self.explainer = None
        self.background_data = None
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        from models.gan_lstm_ae import GANLSTMAutoencoder
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
            model = GANLSTMAutoencoder(784, 128, 20)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def _model_predict(self, x):
        """Model prediction function for SHAP"""
        x_tensor = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            _, error, _, _ = self.model(x_tensor)
        return error.cpu().numpy()
    
    def initialize_explainer(self, background_data: np.ndarray, use_deep: bool = True):
        """
        Initialize SHAP explainer.
        
        Args:
            background_data: Background data for explainer
            use_deep: Whether to try DeepExplainer first
        """
        self.background_data = background_data
        
        if use_deep:
            try:
                # Try DeepExplainer first (faster for neural networks)
                background_tensor = torch.FloatTensor(background_data).to(self.device)
                self.explainer = shap.DeepExplainer(self.model, background_tensor)
                print("SHAP DeepExplainer initialized successfully")
            except Exception as e:
                print(f"DeepExplainer failed: {e}, falling back to KernelExplainer")
                self._initialize_kernel_explainer(background_data)
        else:
            self._initialize_kernel_explainer(background_data)
    
    def _initialize_kernel_explainer(self, background_data: np.ndarray):
        """Initialize KernelExplainer as fallback"""
        self.explainer = shap.KernelExplainer(self._model_predict, background_data)
        print("SHAP KernelExplainer initialized successfully")
    
    def explain(self, sample: np.ndarray, top_k: int = 5) -> dict:
        """
        Explain a single sample using SHAP.
        
        Args:
            sample: Input sample (2D array)
            top_k: Number of top features to return
        
        Returns:
            Dictionary with SHAP values and top features
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Flatten sample if needed
        if sample.ndim == 2:
            sample_flat = sample.flatten().reshape(1, -1)
        else:
            sample_flat = sample.reshape(1, -1)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(sample_flat)
        
        # If shap_values is a list (for multi-output), take the first
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Get feature importances (mean absolute SHAP values)
        feature_importance = np.abs(shap_values[0])
        
        # Get top k features
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        
        top_features = [
            {
                'feature_index': int(idx),
                'importance': float(feature_importance[idx]),
                'shap_value': float(shap_values[0][idx])
            }
            for idx in top_indices
        ]
        
        result = {
            'shap_values': shap_values[0].tolist(),
            'feature_importance': {f'feature_{i}': float(v) for i, v in enumerate(feature_importance)},
            'top_features': top_features,
            'expected_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else None
        }
        
        return result
    
    def explain_batch(self, samples: np.ndarray, top_k: int = 5) -> dict:
        """
        Explain multiple samples.
        
        Args:
            samples: Input samples (3D array: samples, seq_len, features)
            top_k: Number of top features to return
        
        Returns:
            Dictionary with SHAP values for all samples
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Flatten samples
        n_samples = samples.shape[0]
        samples_flat = samples.reshape(n_samples, -1)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(samples_flat)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Get feature importances for each sample
        all_top_features = []
        for i in range(n_samples):
            feature_importance = np.abs(shap_values[i])
            top_indices = np.argsort(feature_importance)[-top_k:][::-1]
            
            top_features = [
                {
                    'feature_index': int(idx),
                    'importance': float(feature_importance[idx]),
                    'shap_value': float(shap_values[i][idx])
                }
                for idx in top_indices
            ]
            all_top_features.append(top_features)
        
        result = {
            'shap_values': shap_values.tolist(),
            'top_features_per_sample': all_top_features
        }
        
        return result


if __name__ == "__main__":
    # Test SHAP explainer
    import sys
    sys.path.append("..")
    from ..models.gan_lstm_ae import GANLSTMAutoencoder
    from sklearn.preprocessing import MinMaxScaler
    
    # Create dummy model
    model = GANLSTMAutoencoder(784, 128, 20)
    Path("models").mkdir(exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, "models/model_test.pt")
    
    # Create dummy scaler
    scaler = MinMaxScaler()
    dummy_data = np.random.randn(100, 784)
    scaler.fit(dummy_data)
    joblib.dump(scaler, "models/scaler_test.pkl")
    
    # Create dummy metadata
    metadata = {'input_dim': 784, 'hidden_dim': 128, 'seq_len': 20}
    joblib.dump(metadata, "models/model_metadata.pkl")
    
    # Test explainer
    explainer = SHAPExplainer("models/model_test.pt", "models/scaler_test.pkl")
    
    background_data = np.random.randn(10, 20 * 784)
    explainer.initialize_explainer(background_data, use_deep=False)
    
    sample = np.random.randn(20, 784)
    result = explainer.explain(sample, top_k=5)
    
    print("SHAP explanation result:")
    print(f"Top features: {result['top_features']}")
