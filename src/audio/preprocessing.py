import numpy as np
import librosa
from typing import Tuple, Dict, Any

from config import AUDIO_SAMPLE_RATE

def remove_dc_offset(waveform: np.ndarray) -> np.ndarray:
    """Removes DC offset by subtracting the mean."""
    return waveform - np.mean(waveform)

def conservative_normalize(waveform: np.ndarray) -> np.ndarray:
    """
    Perform conservative amplitude normalization.
    Scales the audio such that the absolute maximum peak is 0.9.
    Does not apply dynamic range compression.
    """
    peak = np.max(np.abs(waveform))
    if peak > 0:
        return waveform * (0.9 / peak)
    return waveform

def calculate_rms(waveform: np.ndarray) -> float:
    """Calculate the Root Mean Square energy."""
    if len(waveform) == 0:
        return 0.0
    return float(np.sqrt(np.mean(waveform**2)))

def preprocess_audio(waveform: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Preprocesses an audio waveform (DC offset removal, trimming, normalization).
    Runs preprocessing stage for audio.
    
    Args:
        waveform: float32 mono audio array.
        sample_rate: sampling rate of the audio.
        
    Returns:
        processed_waveform
        processing_metadata dictionary
    """
    original_duration = len(waveform) / sample_rate
    peak_before = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
    rms_before = calculate_rms(waveform)
    
    # 1. Remove DC offset
    waveform_dc = remove_dc_offset(waveform)
    
    # 2. Trim silence
    # Using a conservative top_db (e.g. 30 dB below reference) so we don't trim quiet breaths aggressively
    waveform_trimmed, _ = librosa.effects.trim(waveform_dc, top_db=30)
    
    # 3. Conservative normalization
    processed_waveform = conservative_normalize(waveform_trimmed)
    
    processed_duration = len(processed_waveform) / sample_rate
    trimmed_duration = original_duration - processed_duration
    peak_after = float(np.max(np.abs(processed_waveform))) if len(processed_waveform) > 0 else 0.0
    rms_after = calculate_rms(processed_waveform)
    
    metadata = {
        "original_duration": original_duration,
        "processed_duration": processed_duration,
        "trimmed_duration": trimmed_duration,
        "peak_before": peak_before,
        "peak_after": peak_after,
        "rms_before": rms_before,
        "rms_after": rms_after
    }
    
    return processed_waveform, metadata
