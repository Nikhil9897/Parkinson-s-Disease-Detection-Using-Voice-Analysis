import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from typing import List

def build_boosting_pipeline() -> Pipeline:
    """Builds the Gradient Boosting pipeline."""
    return Pipeline([
        ('scaler', RobustScaler()),
        ('boosting', HistGradientBoostingClassifier(random_state=42))
    ])

def train_boosting(X_train: np.ndarray, y_train: np.ndarray, cv_splits: List[tuple]) -> GridSearchCV:
    """
    Trains a Gradient Boosting model using grid search.
    """
    pipeline = build_boosting_pipeline()
    
    param_grid = {
        'boosting__learning_rate': [0.01, 0.1],
        'boosting__max_iter': [100, 200],
        'boosting__max_depth': [3, 5, None]
    }
    
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
