import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, List

def build_svm_pipeline() -> Pipeline:
    """Builds the SVM pipeline with RobustScaler."""
    return Pipeline([
        ('scaler', RobustScaler()),
        ('svc', SVC(kernel='rbf', probability=True, random_state=42))
    ])

def train_svm(X_train: np.ndarray, y_train: np.ndarray, cv_splits: List[tuple]) -> GridSearchCV:
    """
    Trains an SVM model using grid search over a modest parameter space.
    Uses pre-defined cross validation splits to prevent speaker leakage.
    """
    pipeline = build_svm_pipeline()
    
    # Modest hyperparameter search
    param_grid = {
        'svc__C': [0.1, 1.0, 10.0],
        'svc__gamma': ['scale', 'auto', 0.01, 0.1],
        'svc__class_weight': [None, 'balanced']
    }
    
    # We optimize for F1 macro as it is a balanced metric, but user wants primary metric as F1-score for positive class.
    # So we'll use 'f1' as scoring.
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv_splits,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    return grid_search
