import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import DATA_DIR, REPORTS_DIR, MODELS_DIR
from src.training.splits import get_cross_validation_splits
from src.training.metrics import calculate_metrics
from src.models.temporal_model import build_temporal_model

def normalize_temporal_features(X_train: np.ndarray, X_val: np.ndarray = None):
    """
    Normalizes temporal features using statistics derived ONLY from training data.
    Shape: (batch, time, features)
    """
    # Calculate mean and std over batch and time dimensions for each feature
    # Reshape to (batch * time, features) to compute stats
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = np.mean(X_train_flat, axis=0)
    std = np.std(X_train_flat, axis=0)
    
    # Avoid div by zero
    std[std == 0] = 1e-6
    
    X_train_norm = (X_train - mean) / std
    
    if X_val is not None:
        X_val_norm = (X_val - mean) / std
        return X_train_norm, X_val_norm, mean, std
    
    return X_train_norm, mean, std

def train_temporal():
    print("Loading temporal features...")
    temporal_path = CACHE_DIR = DATA_DIR / "cache" / "temporal_features.npz"
    metadata_path = DATA_DIR / "metadata" / "dataset_index.csv"
    
    if not temporal_path.exists() or not metadata_path.exists():
        print("Required files not found. Run cache_temporal.py first.")
        return
        
    df = pd.read_csv(metadata_path)
    data = np.load(temporal_path)
    
    # Build X array maintaining dataframe order
    X_list = []
    y_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        path = row["audio_path"]
        # npz format standardizes paths to forward slashes
        npz_key = path.replace("\\", "/")
        if npz_key in data:
            X_list.append(data[npz_key])
            y_list.append(row["label"])
            valid_indices.append(idx)
            
    df = df.iloc[valid_indices].reset_index(drop=True)
    X = np.array(X_list)
    y = np.array(y_list)
    
    all_fold_metrics = []
    oof_preds = np.zeros(len(X))
    oof_probs = np.zeros(len(X))
    
    splits = list(get_cross_validation_splits(df))
    
    model_name = "CNN-BiLSTM"
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\nTraining {model_name} - Fold {fold+1}/{len(splits)}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Normalize
        X_train, X_val, _, _ = normalize_temporal_features(X_train, X_val)
        
        model = build_temporal_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Use a small learning rate for stability
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save history
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv(REPORTS_DIR / f"training_history_fold_{fold+1}.csv", index=False)
        
        # Predict on validation
        probs = model.predict(X_val).flatten()
        preds = (probs >= 0.5).astype(int)
        
        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds
        
        metrics = calculate_metrics(y_val, preds, probs)
        metrics["model"] = model_name
        metrics["fold"] = fold + 1
        metrics["train_sample_count"] = len(train_idx)
        metrics["validation_sample_count"] = len(val_idx)
        metrics["train_speaker_count"] = df.iloc[train_idx]["speaker_id"].nunique()
        metrics["validation_speaker_count"] = df.iloc[val_idx]["speaker_id"].nunique()
        
        all_fold_metrics.append(metrics)
        
    # Save OOF
    oof_df = df[["filename", "speaker_id", "label"]].copy()
    oof_df.rename(columns={"label": "true_label"}, inplace=True)
    
    fold_map = np.zeros(len(df), dtype=int)
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_map[val_idx] = fold + 1
    oof_df["fold"] = fold_map
    
    oof_df["predicted_probability"] = oof_probs
    oof_df["model"] = model_name
    
    oof_path = REPORTS_DIR / "oof_predictions.csv"
    if os.path.exists(oof_path):
        oof_df.to_csv(oof_path, mode='a', header=False, index=False)
    else:
        oof_df.to_csv(oof_path, index=False)
        
    # Save metrics
    metrics_df = pd.DataFrame(all_fold_metrics)
    cols = ["model", "fold", "accuracy", "roc_auc", "precision", "recall", "f1", "specificity", 
            "train_sample_count", "validation_sample_count", "train_speaker_count", "validation_speaker_count"]
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
    
    metrics_path = REPORTS_DIR / "fold_metrics.csv"
    if os.path.exists(metrics_path):
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_path, index=False)
        
    # Generate model summary
    with open(REPORTS_DIR / "temporal_model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        
    # Final Model Training
    print(f"\nTraining final {model_name} on all data...")
    X_final, mean, std = normalize_temporal_features(X)
    
    final_model = build_temporal_model(input_shape=(X.shape[1], X.shape[2]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    final_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # We use a fixed number of epochs based on average stopping epoch if we wanted, 
    # but here we'll just train for 30 epochs as a safe final pass, or early stop on train loss.
    # We will just train for 40 epochs for the final model.
    final_model.fit(X_final, y, epochs=40, batch_size=16, verbose=1)
    
    final_model.save(MODELS_DIR / "cnn_bilstm.keras")
    
    # Save temporal normalization stats
    np.savez_compressed(MODELS_DIR / "temporal_norm_stats.npz", mean=mean, std=std)
    
    print("Temporal model training complete.")

if __name__ == "__main__":
    train_temporal()
