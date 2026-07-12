import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from config import REPORTS_DIR, MODELS_DIR

def optimize_threshold(oof_df: pd.DataFrame, model_name: str) -> dict:
    """Find the best threshold for F1-score for a given model."""
    df_model = oof_df[oof_df["model"] == model_name].copy()
    
    y_true = df_model["true_label"].values
    y_prob = df_model["predicted_probability"].values
    
    thresholds = np.arange(0.30, 0.71, 0.01)
    best_f1 = -1
    best_thresh = 0.5
    best_recall = -1
    
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        # Calculate manually for speed
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tp + tn) / len(y_true)
        
        results.append({
            "model": model_name,
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "accuracy": acc
        })
        
        # Selection logic: Max F1 -> Max Recall -> Closest to 0.50
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_recall = recall
        elif f1 == best_f1:
            if recall > best_recall:
                best_recall = recall
                best_thresh = t
            elif recall == best_recall:
                if abs(t - 0.5) < abs(best_thresh - 0.5):
                    best_thresh = t
                    
    return {"best_threshold": best_thresh, "best_f1": best_f1, "results": results}

def evaluate_models():
    print("Evaluating models and selecting production model...")
    metrics_path = REPORTS_DIR / "fold_metrics.csv"
    oof_path = REPORTS_DIR / "oof_predictions.csv"
    
    if not metrics_path.exists() or not oof_path.exists():
        print("Metrics or OOF predictions missing.")
        return
        
    metrics_df = pd.read_csv(metrics_path)
    oof_df = pd.read_csv(oof_path)
    
    # 1. Model Comparison Table
    agg_funcs = {
        "accuracy": ["mean", "std"],
        "roc_auc": ["mean", "std"],
        "precision": ["mean"],
        "recall": ["mean"],
        "f1": ["mean", "std"],
        "specificity": ["mean"]
    }
    
    summary = metrics_df.groupby("model").agg(agg_funcs).reset_index()
    # Flatten multi-index columns
    summary.columns = ['_'.join(col).strip('_') if type(col) is tuple else col for col in summary.columns.values]
    
    summary.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print("Model comparison saved.")
    
    # 2. Threshold Optimization
    models = summary["model"].unique()
    all_thresh_results = []
    best_model = None
    best_model_f1 = -1
    best_model_roc = -1
    final_threshold = 0.5
    
    for model in models:
        opt = optimize_threshold(oof_df, model)
        all_thresh_results.extend(opt["results"])
        
        model_mean_f1 = summary[summary["model"] == model]["f1_mean"].values[0]
        model_mean_roc = summary[summary["model"] == model]["roc_auc_mean"].values[0]
        
        # Model Selection logic: Max Mean F1 -> Max Mean ROC-AUC
        if model_mean_f1 > best_model_f1:
            best_model_f1 = model_mean_f1
            best_model_roc = model_mean_roc
            best_model = model
            final_threshold = opt["best_threshold"]
        elif model_mean_f1 == best_model_f1:
            if model_mean_roc > best_model_roc:
                best_model_roc = model_mean_roc
                best_model = model
                final_threshold = opt["best_threshold"]
                
    pd.DataFrame(all_thresh_results).to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    
    print(f"Selected Production Model: {best_model} (Threshold: {final_threshold:.2f})")
    
    # 3. Model Metadata JSON
    metadata_index = pd.read_csv(Path("D:/NeuroVoice-AI/data/metadata/dataset_index.csv"))
    
    # Extract validation metrics for the best model from summary
    best_row = summary[summary["model"] == best_model].iloc[0]
    val_metrics = {
        "mean_accuracy": float(best_row["accuracy_mean"]),
        "std_accuracy": float(best_row["accuracy_std"]),
        "mean_roc_auc": float(best_row["roc_auc_mean"]),
        "mean_precision": float(best_row["precision_mean"]),
        "mean_recall": float(best_row["recall_mean"]),
        "mean_f1": float(best_row["f1_mean"]),
        "std_f1": float(best_row["f1_std"]),
        "mean_specificity": float(best_row["specificity_mean"])
    }
    
    metadata = {
        "project_name": "NeuroVoice AI",
        "project_version": "1.0.0",
        "dataset_source": "figshare_pd_hc_voice_samples",
        "dataset_description": "Explainable Temporal Voice Pattern Screening for Parkinsonian Speech",
        "dataset_file_count": len(metadata_index),
        "healthy_sample_count": len(metadata_index[metadata_index["label"] == 0]),
        "parkinsons_sample_count": len(metadata_index[metadata_index["label"] == 1]),
        "unique_speaker_count": metadata_index["speaker_id"].nunique(),
        "training_date": datetime.now().isoformat(),
        "production_model": best_model,
        "expected_voice_task": "sustained_vowel_a",
        "sample_rate": 16000,
        "input_representation": "40_MFCC_temporal" if best_model == "CNN-BiLSTM" else "20_MFCC_Acoustic_Tabular",
        "validation_strategy": "StratifiedKFold",
        "cross_validation_folds": 5,
        "selected_threshold": float(final_threshold),
        "validation_metrics": val_metrics,
        "random_seed": 42,
        "feature_schema_version": "1.0"
    }
    
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Saved model_metadata.json")

if __name__ == "__main__":
    evaluate_models()
