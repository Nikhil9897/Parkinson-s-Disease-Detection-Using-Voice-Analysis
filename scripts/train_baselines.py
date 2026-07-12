import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from config import DATA_DIR, REPORTS_DIR, MODELS_DIR
from src.training.splits import get_cross_validation_splits
from src.training.metrics import calculate_metrics
from src.models.svm_model import build_svm_pipeline
from src.models.gradient_boosting import build_boosting_pipeline

def train_and_evaluate(model_name: str, pipeline, df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """
    Trains and evaluates a given pipeline using manual cross-validation.
    Returns metrics and OOF predictions.
    """
    splits = list(get_cross_validation_splits(df))
    
    oof_preds = np.zeros(len(X))
    oof_probs = np.zeros(len(X))
    
    fold_metrics_list = []
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"Training {model_name} - Fold {fold+1}/{len(splits)}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit model on train
        pipeline.fit(X_train, y_train)
        
        # Predict on validation
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else preds
        
        oof_preds[val_idx] = preds
        oof_probs[val_idx] = probs
        
        metrics = calculate_metrics(y_val, preds, probs)
        metrics["model"] = model_name
        metrics["fold"] = fold + 1
        metrics["train_sample_count"] = len(train_idx)
        metrics["validation_sample_count"] = len(val_idx)
        metrics["train_speaker_count"] = df.iloc[train_idx]["speaker_id"].nunique()
        metrics["validation_speaker_count"] = df.iloc[val_idx]["speaker_id"].nunique()
        
        fold_metrics_list.append(metrics)
        
    return fold_metrics_list, oof_probs

def train_baselines():
    print("Loading extracted features...")
    features_path = DATA_DIR / "cache" / "acoustic_features.csv"
    metadata_path = DATA_DIR / "metadata" / "dataset_index.csv"
    
    if not features_path.exists() or not metadata_path.exists():
        print("Required files not found. Run extract_features.py first.")
        return
        
    features_df = pd.read_csv(features_path)
    metadata_df = pd.read_csv(metadata_path)
    
    # Merge on audio_path to ensure correct order and labels
    df = pd.merge(metadata_df, features_df, on="audio_path", how="inner")
    
    # Get schema keys to form X matrix
    with open(MODELS_DIR / "acoustic_feature_schema.json", "r") as f:
        schema = json.load(f)
        
    X = df[schema].values
    y = df["label"].values
    
    all_fold_metrics = []
    
    # Initialize OOF dataframe
    oof_df = df[["filename", "speaker_id", "label"]].copy()
    oof_df.rename(columns={"label": "true_label"}, inplace=True)
    
    models_to_run = [
        ("SVM", build_svm_pipeline()),
        ("Gradient Boosting", build_boosting_pipeline())
    ]
    
    # We need fold mapping for OOF predictions
    splits = list(get_cross_validation_splits(df))
    fold_map = np.zeros(len(df), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_map[val_idx] = fold + 1
    oof_df["fold"] = fold_map
    
    for model_name, pipeline in models_to_run:
        fold_metrics, oof_probs = train_and_evaluate(model_name, pipeline, df, X, y)
        all_fold_metrics.extend(fold_metrics)
        
        # Save OOF predictions row format
        model_oof = oof_df.copy()
        model_oof["predicted_probability"] = oof_probs
        model_oof["model"] = model_name
        
        # Append to main file or save intermediate
        oof_path = REPORTS_DIR / "oof_predictions.csv"
        if os.path.exists(oof_path):
            model_oof.to_csv(oof_path, mode='a', header=False, index=False)
        else:
            model_oof.to_csv(oof_path, index=False)
            
        # Train final model on ALL data for saving
        print(f"Training final {model_name} on all data...")
        pipeline.fit(X, y)
        model_filename = "svm_pipeline.joblib" if model_name == "SVM" else "boosting_model.joblib"
        joblib.dump(pipeline, MODELS_DIR / model_filename)
        
    # Save metrics
    metrics_df = pd.DataFrame(all_fold_metrics)
    
    cols = ["model", "fold", "accuracy", "roc_auc", "precision", "recall", "f1", "specificity", 
            "train_sample_count", "validation_sample_count", "train_speaker_count", "validation_speaker_count"]
    
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
    
    # If the file exists (e.g. from temporal model later), we append, but for now we write
    if os.path.exists(REPORTS_DIR / "fold_metrics.csv"):
        metrics_df.to_csv(REPORTS_DIR / "fold_metrics.csv", mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(REPORTS_DIR / "fold_metrics.csv", index=False)
        
    print("Baseline training complete.")
    
if __name__ == "__main__":
    train_baselines()
