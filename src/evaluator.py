"""
Evaluation module for XAI-E-DiD
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from torch.utils.data import DataLoader
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """Evaluator for intrusion detection model"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader) -> dict:
        """Evaluate model on test data"""
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),
            'classification_report': classification_report(all_labels, all_predictions, output_dict=True)
        }
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
        
        # Save results
        self.save_results(metrics)
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        
        return metrics
    
    def save_results(self, metrics: dict):
        """Save evaluation results to JSON"""
        results_dir = Path(self.config.get('paths.results_dir', 'results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = results_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    def plot_confusion_matrix(self, cm: list):
        """Plot and save confusion matrix"""
        results_dir = Path(self.config.get('paths.results_dir', 'results'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plot_path = results_dir / "confusion_matrix.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {plot_path}")
