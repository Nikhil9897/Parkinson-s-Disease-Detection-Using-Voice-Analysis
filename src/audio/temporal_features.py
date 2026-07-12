import numpy as np
import librosa
from config import AUDIO_SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC, MAX_TIME_FRAMES

def extract_temporal_features(waveform: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> np.ndarray:
    """
    Extracts temporal MFCC representation for the deep learning model.
    Expected output shape: (MAX_TIME_FRAMES, N_MFCC)
    """
    if len(waveform) == 0:
        return np.zeros((MAX_TIME_FRAMES, N_MFCC))
        
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Transpose to get (time_frames, n_mfcc)
    mfccs = mfccs.T
    
    time_frames = mfccs.shape[0]
    
    if time_frames > MAX_TIME_FRAMES:
        # Truncate
        mfccs = mfccs[:MAX_TIME_FRAMES, :]
    elif time_frames < MAX_TIME_FRAMES:
        # Post-pad with zeros
        pad_width = MAX_TIME_FRAMES - time_frames
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        
    return mfccs
