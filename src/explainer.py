"""
XAI (Explainable AI) module for model interpretation
"""

import torch
import numpy as np
import shap
import lime
import lime.lime_tabular
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


class XAIExplainer:
    """Explainable AI methods for model interpretation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def explain(self, model_path: str, data_path: str):
        """Generate explanations for model predictions"""
        print("Loading model and data for explanation...")
        
        # Load model
        from .model import ModelFactory
        model = ModelFactory.create_model(self.config)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load data
        from .data_loader import DataLoader
        data_loader = DataLoader(self.config)
        features, labels = data_loader.load_data(data_path)
        features = data_loader.preprocess_data(features, fit=False)
        
        # Generate SHAP explanations
        self._explain_shap(model, features, labels)
        
        # Generate LIME explanations
        self._explain_lime(model, features, labels)
    
    def _explain_shap(self, model, features: np.ndarray, labels: np.ndarray):
        """Generate SHAP explanations"""
        print("\nGenerating SHAP explanations...")
        
        results_dir = Path(self.config.get('paths.results_dir', 'results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Use a subset for explanation
        background_data = features[:100]
        test_data = features[:10]
        
        # Create SHAP explainer
        def model_predict(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()
        
        explainer = shap.KernelExplainer(model_predict, background_data)
        shap_values = explainer.shap_values(test_data)
        
        # Plot summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[1], test_data, show=False)
        plt.savefig(results_dir / "shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP explanations saved to {results_dir}")
    
    def _explain_lime(self, model, features: np.ndarray, labels: np.ndarray):
        """Generate LIME explanations"""
        print("\nGenerating LIME explanations...")
        
        results_dir = Path(self.config.get('paths.results_dir', 'results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        def model_predict(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                logits = model(x_tensor)
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=features,
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
            class_names=['Normal', 'Intrusion'],
            mode='classification'
        )
        
        # Explain first instance
        exp = explainer.explain_instance(
            features[0],
            model_predict,
            num_features=10,
            top_labels=1
        )
        
        # Save explanation
        fig = exp.as_pyplot_figure(label=1)
        fig.savefig(results_dir / "lime_explanation.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"LIME explanations saved to {results_dir}")
