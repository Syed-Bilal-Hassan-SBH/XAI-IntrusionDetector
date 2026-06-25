"""
LIME explainer for XAI-E-DiD
Explains individual predictions using LIME
"""

import torch
import numpy as np
import joblib
from pathlib import Path
import lime
import lime.lime_tabular


class LIMEExplainer:
    """LIME explainer for model interpretation"""
    
    def __init__(self, model_path: str = "models/model.pt", scaler_path: str = "models/scaler.pkl"):
        """
        Initialize LIME explainer.
        
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
        self.feature_names = None
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        from models.gan_lstm_ae import GANLSTMAutoencoder
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
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
        """Model prediction function for LIME"""
        x_tensor = torch.FloatTensor(x).to(self.device)
        with torch.no_grad():
            _, error, _, _ = self.model(x_tensor)
        
        # Convert error to probability-like output
        # Higher error = more likely anomaly
        error_normalized = (error - error.min()) / (error.max() - error.min() + 1e-8)
        
        # Return as binary classification probabilities
        probs = np.column_stack([1 - error_normalized, error_normalized])
        return probs
    
    def initialize_explainer(self, training_data: np.ndarray, feature_names: list = None):
        """
        Initialize LIME explainer.
        
        Args:
            training_data: Training data for explainer
            feature_names: List of feature names
        """
        # Flatten training data if needed
        if training_data.ndim == 3:
            training_data_flat = training_data.reshape(training_data.shape[0], -1)
        else:
            training_data_flat = training_data
        
        # Generate feature names if not provided
        if feature_names is None:
            n_features = training_data_flat.shape[1]
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        self.feature_names = feature_names
        
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data_flat,
            feature_names=feature_names,
            class_names=['normal', 'anomaly'],
            mode='classification',
            discretize_continuous=True
        )
        
        print("LIME explainer initialized successfully")
    
    def explain(self, sample: np.ndarray, num_features: int = 10) -> dict:
        """
        Explain a single sample using LIME.
        
        Args:
            sample: Input sample (2D array)
            num_features: Number of features to include in explanation
        
        Returns:
            Dictionary with LIME explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Flatten sample if needed
        if sample.ndim == 2:
            sample_flat = sample.flatten()
        else:
            sample_flat = sample
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            sample_flat,
            self._model_predict,
            num_features=num_features,
            top_labels=1
        )
        
        # Get explanation for the anomaly class (label 1)
        try:
            explanation = exp.as_list(label=1)
        except:
            explanation = exp.as_list(label=0)
        
        # Convert to dictionary
        feature_importance = {feature: float(importance) for feature, importance in explanation}
        
        # Get top features
        top_features = [
            {
                'feature': feature,
                'importance': float(importance)
            }
            for feature, importance in explanation[:5]
        ]
        
        result = {
            'feature_importance': feature_importance,
            'top_features': top_features,
            'intercept': float(exp.intercept[1]) if len(exp.intercept) > 1 else float(exp.intercept[0]),
            'score': float(exp.score)
        }
        
        return result
    
    def explain_with_visualization(self, sample: np.ndarray, save_path: str = "results/xai/lime_explanation.png", num_features: int = 10):
        """
        Explain sample and save visualization.
        
        Args:
            sample: Input sample
            save_path: Path to save visualization
            num_features: Number of features to show
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Flatten sample
        if sample.ndim == 2:
            sample_flat = sample.flatten()
        else:
            sample_flat = sample
        
        # Generate explanation
        exp = self.explainer.explain_instance(
            sample_flat,
            self._model_predict,
            num_features=num_features,
            top_labels=1
        )
        
        # Save visualization
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            fig = exp.as_pyplot_figure(label=1)
        except:
            fig = exp.as_pyplot_figure(label=0)
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        print(f"LIME visualization saved to {save_path}")
        
        return exp


if __name__ == "__main__":
    # Test LIME explainer
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
    explainer = LIMEExplainer("models/model_test.pt", "models/scaler_test.pkl")
    
    training_data = np.random.randn(50, 20, 784)
    explainer.initialize_explainer(training_data)
    
    sample = np.random.randn(20, 784)
    result = explainer.explain(sample, num_features=10)
    
    print("LIME explanation result:")
    print(f"Top features: {result['top_features']}")
