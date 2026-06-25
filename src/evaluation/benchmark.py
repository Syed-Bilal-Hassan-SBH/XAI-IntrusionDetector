"""
Benchmark module for XAI-E-DiD
Compares baseline vs model performance
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List

from .metrics import EvaluationMetrics


class Benchmark:
    """Benchmark for comparing models"""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize benchmark.
        
        Args:
            results_dir: Directory to save benchmark results
        """
        self.metrics = EvaluationMetrics()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def compare_baseline_vs_model(
        self,
        y_true: np.ndarray,
        baseline_pred: np.ndarray,
        model_pred: np.ndarray,
        baseline_scores: np.ndarray = None,
        model_scores: np.ndarray = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare baseline model vs trained model.
        
        Args:
            y_true: True labels
            baseline_pred: Baseline predictions
            model_pred: Model predictions
            baseline_scores: Baseline prediction scores (optional)
            model_scores: Model prediction scores (optional)
        
        Returns:
            Dictionary comparing both models
        """
        print("=" * 60)
        print("BENCHMARK: BASELINE vs MODEL")
        print("=" * 60)
        
        # Compute baseline metrics
        print("\nComputing baseline metrics...")
        baseline_metrics = self.metrics.compute_all_metrics(
            y_true,
            baseline_pred,
            baseline_scores
        )
        
        # Compute model metrics
        print("Computing model metrics...")
        model_metrics = self.metrics.compute_all_metrics(
            y_true,
            model_pred,
            model_scores
        )
        
        # Compute improvement
        comparison = {}
        for key in baseline_metrics.keys():
            baseline_val = baseline_metrics[key]
            model_val = model_metrics[key]
            improvement = ((model_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
            comparison[key] = {
                'baseline': baseline_val,
                'model': model_val,
                'improvement_pct': improvement
            }
        
        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        for key, values in comparison.items():
            print(f"\n{key}:")
            print(f"  Baseline: {values['baseline']:.4f}")
            print(f"  Model: {values['model']:.4f}")
            print(f"  Improvement: {values['improvement_pct']:+.2f}%")
        
        # Save results
        self._save_comparison(comparison)
        
        return comparison
    
    def compare_multiple_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        scores: Dict[str, np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            y_true: True labels
            predictions: Dictionary of model predictions
            scores: Dictionary of model scores (optional)
        
        Returns:
            DataFrame with comparison results
        """
        print("=" * 60)
        print("BENCHMARK: MULTIPLE MODELS")
        print("=" * 60)
        
        results = []
        
        for model_name, pred in predictions.items():
            print(f"\nEvaluating {model_name}...")
            model_scores = scores[model_name] if scores else None
            
            metrics = self.metrics.compute_all_metrics(y_true, pred, model_scores)
            metrics['model'] = model_name
            results.append(metrics)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.set_index('model')
        
        print("\n" + "=" * 60)
        print("MULTIPLE MODEL COMPARISON")
        print("=" * 60)
        print(df.to_string())
        
        # Save results
        df.to_csv(self.results_dir / "benchmark_comparison.csv")
        print(f"\nBenchmark results saved to {self.results_dir / 'benchmark_comparison.csv'}")
        
        return df
    
    def _save_comparison(self, comparison: Dict[str, Dict[str, float]]):
        """Save comparison results to JSON"""
        with open(self.results_dir / "benchmark_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {self.results_dir / 'benchmark_comparison.json'}")
    
    def generate_benchmark_report(
        self,
        y_true: np.ndarray,
        baseline_pred: np.ndarray,
        model_pred: np.ndarray
    ) -> str:
        """
        Generate a text benchmark report.
        
        Args:
            y_true: True labels
            baseline_pred: Baseline predictions
            model_pred: Model predictions
        
        Returns:
            Text report
        """
        comparison = self.compare_baseline_vs_model(y_true, baseline_pred, model_pred)
        
        report = []
        report.append("=" * 60)
        report.append("BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")
        
        for key, values in comparison.items():
            report.append(f"{key.upper()}")
            report.append(f"  Baseline: {values['baseline']:.4f}")
            report.append(f"  Model: {values['model']:.4f}")
            report.append(f"  Improvement: {values['improvement_pct']:+.2f}%")
            report.append("")
        
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.results_dir / "benchmark_report.txt", 'w') as f:
            f.write(report_text)
        
        print(f"Benchmark report saved to {self.results_dir / 'benchmark_report.txt'}")
        
        return report_text


if __name__ == "__main__":
    # Test benchmark
    benchmark = Benchmark()
    
    # Create dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    baseline_pred = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    model_pred = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    
    # Compare baseline vs model
    comparison = benchmark.compare_baseline_vs_model(y_true, baseline_pred, model_pred)
    
    # Generate report
    report = benchmark.generate_benchmark_report(y_true, baseline_pred, model_pred)
    print(report)
