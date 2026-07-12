"""
src/inference/model_loader.py

Streamlit-aware model loading using @st.cache_resource.
This module separates the heavy loading logic from the predictor
so Streamlit only loads TF/sklearn models once per server session.

Loading order:
  1. Read models/model_metadata.json to find the production model name.
  2. Load the corresponding model artifact (Keras .keras or Joblib .joblib).
  3. Load preprocessing artifacts (norm stats, feature schema).

If a model file is missing, a clear error message is returned.
The application must never fall back to fake predictions.
"""

from pathlib import Path
import json
import numpy as np
from typing import Optional, Dict, Any, Tuple

import joblib

from config import MODELS_DIR


# ---------------------------------------------------------------------------
# Streamlit cache guard
# ---------------------------------------------------------------------------
# We use a try/except so this module can also be imported in non-Streamlit
# contexts (scripts, tests) without errors.
try:
    import streamlit as st
    _cache = st.cache_resource
except Exception:
    # Fallback: plain function (no caching) for non-Streamlit contexts
    def _cache(func):
        return func


@_cache
def load_metadata() -> Optional[Dict[str, Any]]:
    """
    Load and cache model_metadata.json.

    Returns the parsed dict, or None if the file does not exist.
    """
    path = MODELS_DIR / "model_metadata.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


@_cache
def load_production_model():
    """
    Load and cache the production model artifact.

    Returns (model, model_type_str) where model_type_str is one of:
      'svm', 'boosting', 'cnn_bilstm'
    Returns (None, None) if metadata or model file is missing.
    """
    metadata = load_metadata()
    if metadata is None:
        return None, None

    prod = metadata.get("production_model", "")

    if prod == "SVM":
        path = MODELS_DIR / "svm_pipeline.joblib"
        if not path.exists():
            return None, None
        return joblib.load(path), "svm"

    elif prod == "Gradient Boosting":
        path = MODELS_DIR / "boosting_model.joblib"
        if not path.exists():
            return None, None
        return joblib.load(path), "boosting"

    elif prod == "CNN-BiLSTM":
        try:
            import tensorflow as tf
        except ImportError:
            return None, None
        path = MODELS_DIR / "cnn_bilstm.keras"
        if not path.exists():
            return None, None
        model = tf.keras.models.load_model(str(path), compile=False)
        return model, "cnn_bilstm"

    return None, None


@_cache
def load_feature_schema():
    """
    Load and cache the acoustic feature schema (ordered list of feature names).
    Returns a list of str, or None if the file is missing.
    """
    path = MODELS_DIR / "acoustic_feature_schema.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


@_cache
def load_norm_stats() -> Optional[Dict[str, np.ndarray]]:
    """
    Load and cache temporal normalization statistics.
    Required only when the production model is CNN-BiLSTM.
    Returns dict with 'mean' and 'std', or None.
    """
    path = MODELS_DIR / "temporal_norm_stats.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {"mean": data["mean"], "std": data["std"]}


@_cache
def load_training_features() -> Optional[Any]:
    """
    Load and cache training feature data for SHAP background.
    Used by feature_importance.py to build the KernelExplainer background set.
    Returns a numpy array or None.
    """
    cache_path = Path("data/cache/acoustic_features.csv")
    if not cache_path.exists():
        return None
    try:
        import pandas as pd
        schema = load_feature_schema()
        if schema is None:
            return None
        df = pd.read_csv(cache_path)
        available = [c for c in schema if c in df.columns]
        return df[available].values
    except Exception:
        return None
