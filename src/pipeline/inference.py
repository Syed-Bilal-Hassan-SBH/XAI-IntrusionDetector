"""
Inference module for XAI-E-DiD
Loads trained model, accepts new input, returns prediction + explanation
"""

import torch
import numpy as np
import joblib
from pathlib import Path

from pipeline.detect import DetectionPipeline
from pipeline.alert_builder import AlertBuilder
from xai.shap_explainer import SHAPExplainer
from xai.lime_explainer import LIMEExplainer


class InferenceEngine:
    """Inference engine for real-time prediction and explanation"""
    
    def __init__(
        self,
        model_path: str = "models/model.pt",
        scaler_path: str = "models/scaler.pkl",
        use_shap: bool = True,
        use_lime: bool = True
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            use_shap: Whether to use SHAP explanations
            use_lime: Whether to use LIME explanations
        """
        self.detection_pipeline = DetectionPipeline(model_path, scaler_path)
        self.alert_builder = AlertBuilder()
        self.use_shap = use_shap
        self.use_lime = use_lime
        
        # Initialize explainers
        if use_shap:
            try:
                self.shap_explainer = SHAPExplainer(model_path, scaler_path)
            except Exception as e:
                print(f"SHAP explainer initialization failed: {e}")
                self.shap_explainer = None
        
        if use_lime:
            try:
                self.lime_explainer = LIMEExplainer(model_path, scaler_path)
            except Exception as e:
                print(f"LIME explainer initialization failed: {e}")
                self.lime_explainer = None
    
    def predict(self, sample: np.ndarray) -> dict:
        """
        Make prediction on input sample.
        
        Args:
            sample: Input data (2D array)
        
        Returns:
            Prediction dictionary with results
        """
        # Get detection result
        detection_result = self.detection_pipeline.detect_anomaly(sample)
        
        # Build alert
        alert = self.alert_builder.build_alert(
            is_anomaly=detection_result['is_anomaly'],
            confidence=detection_result['confidence'],
            reconstruction_error=detection_result['reconstruction_error'],
            threshold=detection_result['threshold']
        )
        
        return alert
    
    def predict_with_explanation(self, sample: np.ndarray) -> dict:
        """
        Make prediction with explanation.
        
        Args:
            sample: Input data (2D array)
        
        Returns:
            Prediction dictionary with explanations
        """
        # Get detection result
        detection_result = self.detection_pipeline.detect_anomaly(sample)
        
        # Get attention weights if available
        attention_weights = None
        if hasattr(self.detection_pipeline.model, 'attention'):
            sample_clean = self.detection_pipeline.preprocess(sample)
            sequence = self.detection_pipeline.create_sequence(sample_clean)
            with torch.no_grad():
                _, _, attention_weights, _ = self.detection_pipeline.model(
                    torch.FloatTensor(sequence).to(self.detection_pipeline.device)
                )
            attention_weights = attention_weights.cpu().numpy()[0]
        
        # Get feature explanations
        feature_importance = None
        top_features = None
        
        if self.shap_explainer:
            try:
                shap_result = self.shap_explainer.explain(sample)
                feature_importance = shap_result.get('feature_importance')
                top_features = shap_result.get('top_features')
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        if self.lime_explainer and feature_importance is None:
            try:
                lime_result = self.lime_explainer.explain(sample)
                feature_importance = lime_result.get('feature_importance')
                top_features = lime_result.get('top_features')
            except Exception as e:
                print(f"LIME explanation failed: {e}")
        
        # Build alert
        alert = self.alert_builder.build_alert(
            is_anomaly=detection_result['is_anomaly'],
            confidence=detection_result['confidence'],
            reconstruction_error=detection_result['reconstruction_error'],
            threshold=detection_result['threshold'],
            top_features=top_features,
            feature_importance=feature_importance,
            attention_weights=attention_weights
        )
        
        return alert
    
    def set_threshold(self, threshold: float):
        """Set detection threshold"""
        self.detection_pipeline.set_threshold(threshold)
    
    def batch_predict(self, samples: list) -> list:
        """
        Make predictions on multiple samples.
        
        Args:
            samples: List of input samples
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for sample in samples:
            result = self.predict_with_explanation(np.array(sample))
            results.append(result)
        
        return results


if __name__ == "__main__":
    # Test inference engine
    import sys
    sys.path.append("..")
    
    # Create dummy model for testing
    from ..models.gan_lstm_ae import GANLSTMAutoencoder
    import torch
    from sklearn.preprocessing import MinMaxScaler
    
    # Create and save dummy model
    model = GANLSTMAutoencoder(784, 128, 20)
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/model_test.pt")
    
    # Create and save dummy scaler
    scaler = MinMaxScaler()
    dummy_data = np.random.randn(100, 784)
    scaler.fit(dummy_data)
    joblib.dump(scaler, "models/scaler_test.pkl")
    
    # Create dummy metadata
    metadata = {'input_dim': 784, 'hidden_dim': 128, 'seq_len': 20}
    joblib.dump(metadata, "models/model_metadata.pkl")
    
    # Test inference
    engine = InferenceEngine("models/model_test.pt", "models/scaler_test.pkl", use_shap=False, use_lime=False)
    
    # Create dummy sample
    sample = np.random.randn(20, 784)
    result = engine.predict(sample)
    
    print("Prediction result:")
    print(result)
