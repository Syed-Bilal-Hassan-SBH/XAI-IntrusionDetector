"""
Evaluator for XAI-E-DiD
Runs model on test data, computes metrics, saves CSV results
"""

import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import EvaluationMetrics
from pipeline.detect import DetectionPipeline


class Evaluator:
    """Evaluator for anomaly detection model"""
    
    def __init__(
        self,
        model_path: str = "models/model.pt",
        scaler_path: str = "models/scaler.pkl",
        results_dir: str = "results"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            results_dir: Directory to save results
        """
        self.detection_pipeline = DetectionPipeline(model_path, scaler_path)
        self.metrics = EvaluationMetrics()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.results_dir / "metrics").mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = None,
        threshold_multiplier: float = 2.0
    ) -> dict:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features (3D array: samples, seq_len, features)
            y_test: Test labels
            threshold: Detection threshold (if None, compute from data)
            threshold_multiplier: Multiplier for threshold computation
        
        Returns:
            Dictionary of evaluation results
        """
        print("=" * 60)
        print("EVALUATION")
        print("=" * 60)
        
        # Compute threshold if not provided
        if threshold is None:
            print("Computing threshold from test data...")
            normal_samples = X_test[y_test == 0]
            if len(normal_samples) > 0:
                threshold = self.detection_pipeline.compute_threshold(
                    normal_samples[:min(100, len(normal_samples))],
                    multiplier=threshold_multiplier
                )
            else:
                threshold = 0.1
                print(f"Using default threshold: {threshold}")
        
        self.detection_pipeline.set_threshold(threshold)
        
        # Make predictions
        print("Making predictions...")
        predictions = []
        reconstruction_errors = []
        
        for i, sample in enumerate(X_test):
            result = self.detection_pipeline.detect_anomaly(sample)
            predictions.append(1 if result['is_anomaly'] else 0)
            reconstruction_errors.append(result['reconstruction_error'])
        
        predictions = np.array(predictions)
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Compute metrics
        print("Computing metrics...")
        metrics_dict = self.metrics.compute_all_metrics(
            y_test,
            predictions,
            reconstruction_errors
        )
        
        # Compute confusion matrix
        cm = self.metrics.compute_confusion_matrix(y_test, predictions)
        
        # Compute classwise metrics
        classwise_metrics = self.metrics.compute_classwise_metrics(y_test, predictions)
        
        # Print results
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for key, value in metrics_dict.items():
            print(f"{key}: {value:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        print(f"\nClasswise Metrics:")
        for cls, cls_metrics in classwise_metrics.items():
            print(f"  {cls}:")
            for key, value in cls_metrics.items():
                print(f"    {key}: {value:.4f}")
        
        # Save results
        results = {
            'threshold': threshold,
            'metrics': metrics_dict,
            'classwise_metrics': classwise_metrics,
            'confusion_matrix': cm.tolist()
        }
        
        self._save_results(results, reconstruction_errors, y_test, predictions)
        
        return results
    
    def _save_results(
        self,
        results: dict,
        reconstruction_errors: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ):
        """Save evaluation results to files"""
        # Save metrics to CSV
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df.to_csv(self.results_dir / "metrics" / "accuracy.csv", index=False)
        
        # Save F1 scores
        f1_df = pd.DataFrame([results['metrics']])[['f1_score']]
        f1_df.to_csv(self.results_dir / "metrics" / "f1_scores.csv", index=False)
        
        # Save FPR
        fpr_df = pd.DataFrame([results['metrics']])[['fpr']]
        fpr_df.to_csv(self.results_dir / "metrics" / "fpr.csv", index=False)
        
        # Save all metrics as JSON
        with open(self.results_dir / "metrics" / "all_metrics.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrix plot
        self._plot_confusion_matrix(results['confusion_matrix'])
        
        # Save ROC curve
        self._plot_roc_curve(y_true, reconstruction_errors)
        
        print(f"\nResults saved to {self.results_dir}")
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.results_dir / "plots" / "confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix plot saved to {self.results_dir / 'plots' / 'confusion_matrix.png'}")
    
    def _plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray):
        """Plot and save ROC curve"""
        try:
            fpr, tpr, thresholds = self.metrics.compute_roc_curve(y_true, y_scores)
            auc = self.metrics.compute_auc_roc(y_true, y_scores)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(self.results_dir / "plots" / "roc_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"ROC curve saved to {self.results_dir / 'plots' / 'roc_curve.png'}")
        except Exception as e:
            print(f"Could not plot ROC curve: {e}")


if __name__ == "__main__":
    # Test evaluator
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
    
    # Test evaluator
    evaluator = Evaluator("models/model_test.pt", "models/scaler_test.pt")
    
    # Create dummy test data
    X_test = np.random.randn(50, 20, 784)
    y_test = np.random.randint(0, 2, 50)
    
    results = evaluator.evaluate(X_test, y_test)
    print("\nEvaluation test completed")
