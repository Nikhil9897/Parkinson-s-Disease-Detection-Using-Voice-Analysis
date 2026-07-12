import numpy as np
import librosa
from typing import Dict, List

from config import AUDIO_SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC

def extract_acoustic_features(waveform: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> Dict[str, float]:
    """
    Extract deterministic acoustic features for the classical ML representation.
    """
    features = {}
    
    if len(waveform) == 0:
        return features

    # 1. Pitch Features
    # f0, voiced_flag, voiced_probs = librosa.pyin
    f0, voiced_flag, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate,
        frame_length=2048,
        hop_length=HOP_LENGTH,
        fill_na=None
    )
    
    voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else []
    
    if len(voiced_f0) > 0:
        features["f0_mean"] = float(np.mean(voiced_f0))
        features["f0_median"] = float(np.median(voiced_f0))
        features["f0_std"] = float(np.std(voiced_f0))
        features["f0_min"] = float(np.min(voiced_f0))
        features["f0_max"] = float(np.max(voiced_f0))
        features["f0_range"] = float(np.max(voiced_f0) - np.min(voiced_f0))
        q75, q25 = np.percentile(voiced_f0, [75, 25])
        features["f0_iqr"] = float(q75 - q25)
    else:
        features["f0_mean"] = 0.0
        features["f0_median"] = 0.0
        features["f0_std"] = 0.0
        features["f0_min"] = 0.0
        features["f0_max"] = 0.0
        features["f0_range"] = 0.0
        features["f0_iqr"] = 0.0

    features["voiced_frame_ratio"] = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0

    # Optional Jitter / Shimmer approx calculation
    # Using consecutive pitch periods
    if len(voiced_f0) > 1:
        # absolute differences between consecutive f0 periods
        f0_periods = 1.0 / voiced_f0
        period_diffs = np.abs(np.diff(f0_periods))
        mean_period = np.mean(f0_periods)
        features["estimated_local_jitter"] = float(np.mean(period_diffs) / mean_period) if mean_period > 0 else 0.0
    else:
        features["estimated_local_jitter"] = 0.0
        
    # Shimmer is usually on peak amplitude per cycle, which is harder. We will skip shimmer to avoid indefensible values or we can do RMS diffs over voiced frames.
    # We will use RMS diffs on voiced frames as a proxy if requested, but better to set to 0 or omit if not defensible.
    features["estimated_local_shimmer"] = 0.0

    # 2. Energy Features
    rms = librosa.feature.rms(y=waveform, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))
    features["rms_median"] = float(np.median(rms))
    q75, q25 = np.percentile(rms, [75, 25])
    features["rms_iqr"] = float(q75 - q25)

    # 3. Temporal Features
    zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # 4. Spectral Features
    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    flatness = librosa.feature.spectral_flatness(y=waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    features["spectral_flatness_mean"] = float(np.mean(flatness))
    features["spectral_flatness_std"] = float(np.std(flatness))

    # 5. MFCC Features (20 coefficients for classical model)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    for i in range(20):
        features[f"mfcc_{i+1:02d}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1:02d}_std"] = float(np.std(mfcc[i]))

    # 6. Delta MFCC
    mfcc_delta = librosa.feature.delta(mfcc)
    for i in range(20):
        features[f"mfcc_delta_{i+1:02d}_mean"] = float(np.mean(mfcc_delta[i]))
        features[f"mfcc_delta_{i+1:02d}_std"] = float(np.std(mfcc_delta[i]))

    return features

def get_acoustic_feature_schema() -> List[str]:
    """Returns the ordered list of acoustic feature names."""
    # Build dummy to get keys
    dummy_wav = np.zeros(AUDIO_SAMPLE_RATE)
    # just create keys directly to avoid slow pyin on dummy
    keys = [
        "f0_mean", "f0_median", "f0_std", "f0_min", "f0_max", "f0_range", "f0_iqr", "voiced_frame_ratio",
        "estimated_local_jitter", "estimated_local_shimmer",
        "rms_mean", "rms_std", "rms_median", "rms_iqr",
        "zcr_mean", "zcr_std",
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_flatness_mean", "spectral_flatness_std"
    ]
    for i in range(20):
        keys.append(f"mfcc_{i+1:02d}_mean")
        keys.append(f"mfcc_{i+1:02d}_std")
    for i in range(20):
        keys.append(f"mfcc_delta_{i+1:02d}_mean")
        keys.append(f"mfcc_delta_{i+1:02d}_std")
    return keys
