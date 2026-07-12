import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
    """
    Calculate classification metrics safely.
    """
    metrics = {}
    
    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    # ROC AUC
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = 0.5
    else:
        metrics["roc_auc"] = 0.5
        
    # Precision, Recall, F1
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Sensitivity & Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # For storage, it's easier to keep just the raw counts for confusion matrix
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    
    return metrics
