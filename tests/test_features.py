import numpy as np
from src.audio.acoustic_features import extract_acoustic_features
from src.audio.temporal_features import extract_temporal_features

def test_extract_acoustic_features():
    waveform = np.random.randn(3 * 16000).astype(np.float32)
    features = extract_acoustic_features(waveform, sample_rate=16000)
    
    assert isinstance(features, dict)
    assert "f0_mean" in features
    assert "zcr_mean" in features

def test_extract_temporal_features():
    waveform = np.random.randn(3 * 16000).astype(np.float32)
    mfcc_seq = extract_temporal_features(waveform, sample_rate=16000)
    
    assert isinstance(mfcc_seq, np.ndarray)
    assert mfcc_seq.ndim == 2
    # Should be (time_steps, feature_dim) or padded to (max_len, feature_dim)
    assert mfcc_seq.shape[1] == 40
