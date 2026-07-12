"""
src/explainability/feature_importance.py

SHAP-based feature importance for classical ML models (SVM, Gradient Boosting).
Uses KernelExplainer for SVM and TreeExplainer for gradient boosting.

IMPORTANT: SHAP values indicate model influence direction and magnitude.
They do not represent clinical causality, diagnostic markers, or disease causes.
Use the label: "Model influence" — not "Disease cause" or "Clinical marker".
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

# Human-readable labels for internal feature names
FEATURE_LABELS: Dict[str, str] = {
    "f0_mean": "Mean F0 (pitch)",
    "f0_median": "Median F0",
    "f0_std": "F0 variability (std)",
    "f0_range": "F0 range",
    "f0_iqr": "F0 IQR",
    "f0_min": "F0 minimum",
    "f0_max": "F0 maximum",
    "voiced_frame_ratio": "Voiced frame ratio",
    "estimated_local_jitter": "Estimated jitter (approx.)",
    "rms_mean": "Mean RMS energy",
    "rms_std": "RMS energy variability",
    "rms_median": "Median RMS energy",
    "rms_iqr": "RMS energy IQR",
    "zcr_mean": "Mean zero-crossing rate",
    "zcr_std": "ZCR variability",
    "spectral_centroid_mean": "Spectral centroid (mean)",
    "spectral_centroid_std": "Spectral centroid (std)",
    "spectral_bandwidth_mean": "Spectral bandwidth (mean)",
    "spectral_bandwidth_std": "Spectral bandwidth (std)",
    "spectral_rolloff_mean": "Spectral rolloff (mean)",
    "spectral_rolloff_std": "Spectral rolloff (std)",
    "spectral_flatness_mean": "Spectral flatness (mean)",
    "spectral_flatness_std": "Spectral flatness (std)",
}

# Generate MFCC labels
for i in range(1, 21):
    FEATURE_LABELS[f"mfcc_{i:02d}_mean"] = f"MFCC {i} mean"
    FEATURE_LABELS[f"mfcc_{i:02d}_std"] = f"MFCC {i} variability"
    FEATURE_LABELS[f"mfcc_delta_{i:02d}_mean"] = f"ΔMFCC {i} mean"
    FEATURE_LABELS[f"mfcc_delta_{i:02d}_std"] = f"ΔMFCC {i} variability"


def get_feature_label(feature_name: str) -> str:
    """Return a human-readable label for an internal feature name."""
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def compute_shap_values(
    model,
    feature_vector: np.ndarray,
    background_data: Optional[np.ndarray] = None,
    model_type: str = "boosting",
    n_background: int = 50,
) -> Optional[Dict[str, Any]]:
    """
    Compute SHAP values for a single prediction.

    Parameters
    ----------
    model : sklearn pipeline or estimator
        Trained production model.
    feature_vector : np.ndarray, shape (1, n_features)
        Feature vector for the sample being explained.
    background_data : np.ndarray or None
        Background dataset for KernelExplainer. Required for SVM.
        For tree models, this is not needed.
    model_type : str
        'boosting' or 'svm'.
    n_background : int
        Number of background samples for KernelExplainer.

    Returns
    -------
    dict with keys: shap_values, base_value
    Or None if SHAP is unavailable or fails.
    """
    if not SHAP_AVAILABLE:
        return None

    try:
        if model_type == "boosting":
            # HistGradientBoostingClassifier — use KernelExplainer
            # (TreeExplainer does not support HistGradientBoosting)
            if background_data is not None:
                bg = shap.kmeans(background_data, min(n_background, len(background_data)))
            else:
                bg = shap.kmeans(feature_vector, 1)

            def predict_proba_pos(X):
                return model.predict_proba(X)[:, 1]

            explainer = shap.KernelExplainer(predict_proba_pos, bg)
            shap_vals = explainer.shap_values(feature_vector, nsamples=100, silent=True)
            base_value = float(explainer.expected_value)

        elif model_type == "svm":
            if background_data is not None:
                bg = shap.kmeans(background_data, min(n_background, len(background_data)))
            else:
                bg = shap.kmeans(feature_vector, 1)

            def predict_proba_pos(X):
                return model.predict_proba(X)[:, 1]

            explainer = shap.KernelExplainer(predict_proba_pos, bg)
            shap_vals = explainer.shap_values(feature_vector, nsamples=100, silent=True)
            base_value = float(explainer.expected_value)
        else:
            return None

        return {
            "shap_values": shap_vals[0] if hasattr(shap_vals, "__len__") else shap_vals,
            "base_value": base_value,
        }

    except Exception as e:
        # SHAP failure is non-fatal. Caller should handle gracefully.
        return {"error": str(e)}


def build_top_influences(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Build a DataFrame of top model influences for display.

    Returns DataFrame with columns:
        feature, label, shap_value, direction
    Sorted by absolute SHAP value descending.
    """
    records = []
    for i, name in enumerate(feature_names):
        val = float(shap_values[i])
        records.append({
            "feature": name,
            "label": get_feature_label(name),
            "shap_value": val,
            "abs_shap": abs(val),
            "direction": "↑ Increases" if val > 0 else "↓ Decreases",
        })

    df = pd.DataFrame(records)
    df = df.sort_values("abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    return df
