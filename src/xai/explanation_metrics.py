"""
Explanation metrics for XAI-E-DiD
Implements fidelity, consistency, and latency metrics
"""

import time
import numpy as np
from typing import Dict, List, Any


class ExplanationMetrics:
    """Metrics for evaluating explanation quality"""
    
    def __init__(self):
        self.metrics_history = []
    
    def compute_fidelity(self, original_prediction: float, perturbed_predictions: List[float]) -> float:
        """
        Compute fidelity metric - how stable predictions are under perturbation.
        
        Args:
            original_prediction: Original model prediction
            perturbed_predictions: List of predictions on perturbed data
        
        Returns:
            Fidelity score (higher is better)
        """
        if not perturbed_predictions:
            return 0.0
        
        # Compute mean absolute deviation
        deviations = [abs(original_prediction - pred) for pred in perturbed_predictions]
        mean_deviation = np.mean(deviations)
        
        # Fidelity is inverse of deviation
        fidelity = 1.0 / (1.0 + mean_deviation)
        
        return float(fidelity)
    
    def compute_consistency(
        self,
        explanation1: Dict[str, float],
        explanation2: Dict[str, float],
        top_k: int = 5
    ) -> float:
        """
        Compute consistency metric - how similar explanations are for similar inputs.
        
        Args:
            explanation1: First explanation (feature importance dict)
            explanation2: Second explanation (feature importance dict)
            top_k: Number of top features to compare
        
        Returns:
            Consistency score (higher is better)
        """
        # Get top k features from each explanation
        top_features1 = sorted(explanation1.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        top_features2 = sorted(explanation2.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        
        # Extract feature names
        feature_names1 = set(f[0] for f in top_features1)
        feature_names2 = set(f[0] for f in top_features2)
        
        # Compute overlap
        overlap = len(feature_names1.intersection(feature_names2))
        
        # Consistency is ratio of overlap to top_k
        consistency = overlap / top_k if top_k > 0 else 0.0
        
        return float(consistency)
    
    def compute_latency(self, explanation_func, *args, **kwargs) -> float:
        """
        Compute latency metric - time taken to generate explanation.
        
        Args:
            explanation_func: Function that generates explanation
            *args: Arguments for explanation function
            **kwargs: Keyword arguments for explanation function
        
        Returns:
            Latency in seconds
        """
        start_time = time.time()
        result = explanation_func(*args, **kwargs)
        end_time = time.time()
        
        latency = end_time - start_time
        
        return float(latency)
    
    def compute_completeness(
        self,
        explanation: Dict[str, float],
        total_features: int
    ) -> float:
        """
        Compute completeness metric - what fraction of features are explained.
        
        Args:
            explanation: Feature importance dictionary
            total_features: Total number of features
        
        Returns:
            Completeness score (higher is better)
        """
        explained_features = len(explanation)
        completeness = explained_features / total_features if total_features > 0 else 0.0
        
        return float(completeness)
    
    def compute_sparsity(self, explanation: Dict[str, float], threshold: float = 0.01) -> float:
        """
        Compute sparsity metric - how many features have significant importance.
        
        Args:
            explanation: Feature importance dictionary
            threshold: Threshold for considering a feature significant
        
        Returns:
            Sparsity score (lower is more sparse)
        """
        significant_features = sum(1 for v in explanation.values() if abs(v) > threshold)
        total_features = len(explanation)
        
        sparsity = significant_features / total_features if total_features > 0 else 0.0
        
        return float(sparsity)
    
    def evaluate_explanation(
        self,
        original_prediction: float,
        perturbed_predictions: List[float],
        explanation: Dict[str, float],
        total_features: int,
        explanation_func=None,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute all explanation metrics.
        
        Args:
            original_prediction: Original model prediction
            perturbed_predictions: Predictions on perturbed data
            explanation: Feature importance dictionary
            total_features: Total number of features
            explanation_func: Function to measure latency (optional)
            *args: Arguments for explanation function
            **kwargs: Keyword arguments for explanation function
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'fidelity': self.compute_fidelity(original_prediction, perturbed_predictions),
            'completeness': self.compute_completeness(explanation, total_features),
            'sparsity': self.compute_sparsity(explanation)
        }
        
        # Compute latency if explanation function is provided
        if explanation_func is not None:
            metrics['latency'] = self.compute_latency(explanation_func, *args, **kwargs)
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average of all metrics computed so far.
        
        Returns:
            Dictionary of average metrics
        """
        if not self.metrics_history:
            return {}
        
        avg_metrics = {}
        
        for key in self.metrics_history[0].keys():
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                avg_metrics[key] = float(np.mean(values))
        
        return avg_metrics
    
    def reset(self):
        """Reset metrics history"""
        self.metrics_history = []


if __name__ == "__main__":
    # Test explanation metrics
    metrics = ExplanationMetrics()
    
    # Test fidelity
    original_pred = 0.8
    perturbed_preds = [0.75, 0.82, 0.78, 0.79]
    fidelity = metrics.compute_fidelity(original_pred, perturbed_preds)
    print(f"Fidelity: {fidelity:.4f}")
    
    # Test consistency
    exp1 = {'feature_1': 0.5, 'feature_2': 0.3, 'feature_3': 0.2}
    exp2 = {'feature_1': 0.4, 'feature_2': 0.35, 'feature_4': 0.25}
    consistency = metrics.compute_consistency(exp1, exp2, top_k=3)
    print(f"Consistency: {consistency:.4f}")
    
    # Test latency
    def dummy_explanation():
        time.sleep(0.1)
        return {'feature_1': 0.5}
    
    latency = metrics.compute_latency(dummy_explanation)
    print(f"Latency: {latency:.4f}s")
    
    # Test completeness
    explanation = {'feature_1': 0.5, 'feature_2': 0.3}
    completeness = metrics.compute_completeness(explanation, total_features=10)
    print(f"Completeness: {completeness:.4f}")
    
    # Test sparsity
    sparsity = metrics.compute_sparsity(explanation)
    print(f"Sparsity: {sparsity:.4f}")
    
    print("\nAll metrics tests completed")
