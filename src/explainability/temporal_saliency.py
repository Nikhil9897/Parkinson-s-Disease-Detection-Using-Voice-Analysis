"""
src/explainability/temporal_saliency.py

Temporal attention weight extraction for the CNN-BiLSTM model.

The attention layer (src/models/attention.py) produces a weighted
representation of the hidden states over time. This module builds
a secondary Keras model that exposes those attention weights
so they can be visualized in the UI.

DISCLAIMER:
Attention weights indicate which time regions had stronger influence
on the model output — they do NOT establish clinical causality
or identify pathological events. Use the label:
    "Model time-step importance" or "Temporal model attention"
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any


def build_attention_extractor(full_model):
    """
    Build a secondary model that outputs the attention weights
    from the TemporalAttention layer in addition to the final prediction.

    Parameters
    ----------
    full_model : tf.keras.Model
        The trained CNN-BiLSTM model containing a TemporalAttention layer.

    Returns
    -------
    extractor : tf.keras.Model or None
        A model with outputs = [final_prediction, attention_weights].
        Returns None if the attention layer is not found or TF is unavailable.
    """
    try:
        import tensorflow as tf
        from keras import Model

        attention_layer = None
        for layer in full_model.layers:
            if "temporal_attention" in layer.name.lower():
                attention_layer = layer
                break

        if attention_layer is None:
            return None

        # Build a sub-model that returns the attention weight intermediate output.
        # We need the layer input to create a raw attention score model.
        # Because our TemporalAttention.call() does not expose alpha explicitly,
        # we reconstruct it by re-running the attention math.
        # Simpler approach: use the BiLSTM output as attention input, then run the
        # attention weights formula manually.

        bilstm_output = None
        for layer in full_model.layers:
            if "bidirectional" in layer.name.lower():
                bilstm_output = layer.output
                break

        if bilstm_output is None:
            return None

        # Rebuild attention weights from the stored layer weights
        W = attention_layer.W
        b = attention_layer.b

        # attention_input: (batch, time, hidden)
        attn_input = bilstm_output
        e = tf.tanh(tf.matmul(attn_input, W) + b)      # (batch, time, 1)
        alpha = tf.nn.softmax(e, axis=1)                 # (batch, time, 1)
        alpha_squeezed = tf.squeeze(alpha, axis=-1)      # (batch, time)

        extractor = Model(
            inputs=full_model.input,
            outputs=[full_model.output, alpha_squeezed],
            name="attention_extractor",
        )
        return extractor

    except Exception:
        return None


def get_temporal_importance(
    full_model,
    mfcc_batch: np.ndarray,
    hop_length: int = 512,
    sample_rate: int = 16000,
) -> Optional[Dict[str, Any]]:
    """
    Extract per-frame attention weights and map them to time intervals.

    Parameters
    ----------
    full_model : tf.keras.Model
        Trained CNN-BiLSTM model.
    mfcc_batch : np.ndarray, shape (1, time_frames, n_mfcc)
        Normalized MFCC input batch.
    hop_length : int
        Hop length used during MFCC extraction.
    sample_rate : int
        Audio sample rate.

    Returns
    -------
    dict with keys:
        attention_weights : np.ndarray  (time_frames,)
        frame_times : np.ndarray        (time_frames,)
        top_intervals : list of (start_sec, end_sec, weight)
    Or None if extraction fails.
    """
    try:
        import numpy as np
        extractor = build_attention_extractor(full_model)
        if extractor is None:
            return None

        pred, alpha = extractor.predict(mfcc_batch, verbose=0)
        attention_weights = alpha[0]  # (time_frames,)

        n_frames = len(attention_weights)
        frame_times = np.arange(n_frames) * hop_length / sample_rate

        # Find top influential intervals (contiguous high-attention regions)
        top_intervals = _extract_top_intervals(attention_weights, frame_times, n_top=3)

        return {
            "attention_weights": attention_weights,
            "frame_times": frame_times,
            "top_intervals": top_intervals,
            "prediction": float(pred[0, 0]),
        }
    except Exception:
        return None


def _extract_top_intervals(
    weights: np.ndarray,
    times: np.ndarray,
    n_top: int = 3,
    threshold_pct: float = 0.70,
) -> List[Tuple[float, float, float]]:
    """
    Find contiguous time intervals where attention weight exceeds a threshold.

    Returns a list of (start_sec, end_sec, mean_weight) tuples,
    sorted by mean_weight descending.
    """
    if len(weights) == 0 or len(times) == 0:
        return []

    threshold = float(np.quantile(weights, threshold_pct))
    above = weights >= threshold

    intervals = []
    in_region = False
    start_idx = 0

    for i, flag in enumerate(above):
        if flag and not in_region:
            in_region = True
            start_idx = i
        elif not flag and in_region:
            in_region = False
            region_weights = weights[start_idx:i]
            intervals.append((
                round(float(times[start_idx]), 3),
                round(float(times[i - 1]), 3),
                round(float(np.mean(region_weights)), 4),
            ))

    if in_region:
        region_weights = weights[start_idx:]
        intervals.append((
            round(float(times[start_idx]), 3),
            round(float(times[-1]), 3),
            round(float(np.mean(region_weights)), 4),
        ))

    intervals.sort(key=lambda x: x[2], reverse=True)
    return intervals[:n_top]
