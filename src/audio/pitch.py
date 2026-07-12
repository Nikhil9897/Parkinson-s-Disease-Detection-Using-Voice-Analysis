"""
src/audio/pitch.py

Dedicated pitch extraction helper using librosa.pyin.
All pitch-related calculations are centralized here so both
acoustic_features.py and charts.py can share the same implementation.
"""

import numpy as np
import librosa
from typing import Tuple, Dict, Any

from config import AUDIO_SAMPLE_RATE, HOP_LENGTH


PYIN_FMIN = librosa.note_to_hz('C2')   # ~65 Hz
PYIN_FMAX = librosa.note_to_hz('C7')   # ~2093 Hz
PYIN_FRAME_LENGTH = 2048


def extract_f0(
    waveform: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run pyin pitch tracking and return raw arrays.

    Returns
    -------
    f0 : np.ndarray
        Pitch estimate per frame. NaN where unvoiced.
    voiced_flag : np.ndarray of bool
        True for frames estimated as voiced.
    voiced_probs : np.ndarray
        Per-frame voicing probability from pyin.
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        waveform,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=sample_rate,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=hop_length,
        fill_na=None,
    )
    return f0, voiced_flag, voiced_probs


def compute_f0_statistics(
    waveform: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
) -> Dict[str, Any]:
    """
    Extract pitch statistics using only valid voiced frames.
    Unvoiced (NaN) frames are excluded from all statistics.

    Returns a dict containing:
        f0_mean, f0_median, f0_std, f0_min, f0_max,
        f0_range, f0_iqr, voiced_frame_ratio,
        f0_raw (ndarray), voiced_flag (ndarray), time_axis (ndarray)
    """
    f0, voiced_flag, _ = extract_f0(waveform, sample_rate, hop_length)

    time_axis = librosa.frames_to_time(
        np.arange(len(f0)), sr=sample_rate, hop_length=hop_length
    )

    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

    voiced_frame_ratio = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0

    if len(voiced_f0) > 0:
        q75, q25 = np.percentile(voiced_f0, [75, 25])
        stats: Dict[str, Any] = {
            "f0_mean": float(np.mean(voiced_f0)),
            "f0_median": float(np.median(voiced_f0)),
            "f0_std": float(np.std(voiced_f0)),
            "f0_min": float(np.min(voiced_f0)),
            "f0_max": float(np.max(voiced_f0)),
            "f0_range": float(np.max(voiced_f0) - np.min(voiced_f0)),
            "f0_iqr": float(q75 - q25),
        }
    else:
        stats = {
            "f0_mean": 0.0, "f0_median": 0.0, "f0_std": 0.0,
            "f0_min": 0.0, "f0_max": 0.0, "f0_range": 0.0, "f0_iqr": 0.0,
        }

    stats["voiced_frame_ratio"] = voiced_frame_ratio
    stats["f0_raw"] = f0
    stats["voiced_flag"] = voiced_flag
    stats["time_axis"] = time_axis

    return stats


def estimate_local_jitter(voiced_f0: np.ndarray) -> float:
    """
    Approximate local jitter as the mean absolute difference between
    consecutive pitch periods, normalized by the mean period.

    This is a Librosa-derived approximation.
    It is NOT equivalent to validated clinical MDVP jitter measurements.
    """
    if len(voiced_f0) < 2:
        return 0.0
    f0_periods = 1.0 / voiced_f0
    period_diffs = np.abs(np.diff(f0_periods))
    mean_period = np.mean(f0_periods)
    return float(np.mean(period_diffs) / mean_period) if mean_period > 0 else 0.0
