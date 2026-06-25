"""
Evaluation metrics for XAI-E-DiD
Implements accuracy, precision, recall, F1, and false positive rate
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from typing import Dict, List, Tuple


class EvaluationMetrics:
    """Compute evaluation metrics for anomaly detection"""
    
    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy"""
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def compute_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Compute precision"""
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def compute_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Compute recall"""
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
        """Compute F1 score"""
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute False Positive Rate.
        FPR = FP / (FP + TN)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(fpr)
    
    @staticmethod
    def compute_tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute True Positive Rate (Recall).
        TPR = TP / (TP + FN)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(tpr)
    
    @staticmethod
    def compute_tnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute True Negative Rate (Specificity).
        TNR = TN / (TN + FP)
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(tnr)
    
    @staticmethod
    def compute_auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUC-ROC score"""
        try:
            return float(roc_auc_score(y_true, y_scores))
        except ValueError:
            return 0.0
    
    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray = None,
        minority_classes: list = None
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores (optional, for AUC-ROC)
            minority_classes: List of minority class labels to compute F1 for
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {
            'accuracy': EvaluationMetrics.compute_accuracy(y_true, y_pred),
            'precision': EvaluationMetrics.compute_precision(y_true, y_pred),
            'recall': EvaluationMetrics.compute_recall(y_true, y_pred),
            'f1_score': EvaluationMetrics.compute_f1(y_true, y_pred),
            'fpr': EvaluationMetrics.compute_fpr(y_true, y_pred),
            'tpr': EvaluationMetrics.compute_tpr(y_true, y_pred),
            'tnr': EvaluationMetrics.compute_tnr(y_true, y_pred)
        }
        
        if y_scores is not None:
            metrics['auc_roc'] = EvaluationMetrics.compute_auc_roc(y_true, y_scores)
        
        # Compute minority class F1 scores
        if minority_classes:
            for cls in minority_classes:
                y_true_binary = (y_true == cls).astype(int)
                y_pred_binary = (y_pred == cls).astype(int)
                minority_f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics[f'minority_class_{int(cls)}_f1'] = minority_f1
        
        return metrics
    
    @staticmethod
    def compute_classwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each class.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary of metrics per class
        """
        classes = np.unique(y_true)
        classwise_metrics = {}
        
        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            classwise_metrics[f'class_{int(cls)}'] = {
                'precision': float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
                'recall': float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
                'f1_score': float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
            }
        
        return classwise_metrics
    
    @staticmethod
    def compute_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        return roc_curve(y_true, y_scores)


if __name__ == "__main__":
    # Test evaluation metrics
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.6, 0.7])
    
    metrics = EvaluationMetrics()
    
    # Compute all metrics
    all_metrics = metrics.compute_all_metrics(y_true, y_pred, y_scores)
    
    print("Evaluation Metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Compute classwise metrics
    classwise = metrics.compute_classwise_metrics(y_true, y_pred)
    print("\nClasswise Metrics:")
    for cls, cls_metrics in classwise.items():
        print(f"  {cls}:")
        for key, value in cls_metrics.items():
            print(f"    {key}: {value:.4f}")
    
    # Compute confusion matrix
    cm = metrics.compute_confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
