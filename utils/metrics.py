"""
Evaluation metrics for TB detection
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_metrics(y_true, y_pred, y_probs=None):
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (for AUC-ROC)
    
    Returns:
        dict with all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity and Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:  # Binary classification
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC (requires probabilities)
    if y_probs is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['auc_roc'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Pretty print metrics."""
    print(f"\n{title}:")
    print(f"  Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:   {metrics.get('precision', 0):.4f}")
    print(f"  Recall:      {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:    {metrics.get('f1', 0):.4f}")
    print(f"  AUC-ROC:     {metrics.get('auc_roc', 0):.4f}")
    print(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
    print(f"  Specificity: {metrics.get('specificity', 0):.4f}")
    
    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    [[TN={cm[0][0]:3d}, FP={cm[0][1]:3d}]")
        print(f"     [FN={cm[1][0]:3d}, TP={cm[1][1]:3d}]]")
