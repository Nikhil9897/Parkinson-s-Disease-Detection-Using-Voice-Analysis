import numpy as np
import librosa
from typing import Dict, Any

from config import AUDIO_SAMPLE_RATE, MIN_AUDIO_DURATION, MAX_AUDIO_DURATION

def assess_audio_quality(waveform: np.ndarray, sample_rate: int = AUDIO_SAMPLE_RATE) -> Dict[str, Any]:
    """
    Assess technical quality of the audio signal.
    Runs technical metrics validation.
    """
    metrics = {}
    warnings = []
    blocking_errors = []
    
    # Duration
    duration = len(waveform) / sample_rate
    metrics["duration"] = float(duration)
    
    if duration < MIN_AUDIO_DURATION:
        blocking_errors.append(f"Recording is too short. Minimum required is {MIN_AUDIO_DURATION}s.")
    if duration > MAX_AUDIO_DURATION:
        warnings.append(f"Recording is longer than {MAX_AUDIO_DURATION}s and will be truncated.")
        
    # Amplitude metrics
    rms = float(np.sqrt(np.mean(waveform**2))) if len(waveform) > 0 else 0.0
    peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
    metrics["rms_energy"] = rms
    metrics["peak_amplitude"] = peak
    
    if rms < 0.001:
        blocking_errors.append("Audio is nearly silent.")
    elif rms < 0.01:
        warnings.append("Audio volume is very low.")
        
    # Clipping ratio (samples very close to 1.0 or -1.0, assuming normalization beforehand or from source)
    clipping_threshold = 0.99
    clipping_samples = np.sum(np.abs(waveform) >= clipping_threshold)
    clipping_ratio = float(clipping_samples / len(waveform)) if len(waveform) > 0 else 0.0
    metrics["clipping_ratio"] = clipping_ratio
    
    if clipping_ratio > 0.05:
        blocking_errors.append("Severe clipping detected.")
    elif clipping_ratio > 0.01:
        warnings.append("Minor clipping detected.")
        
    # Silence ratio
    # Frames with RMS energy < 10% of mean RMS
    frame_length = 2048
    hop_length = 512
    if len(waveform) >= frame_length:
        rms_curve = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
        mean_rms = np.mean(rms_curve)
        silence_frames = np.sum(rms_curve < (0.1 * mean_rms))
        silence_ratio = float(silence_frames / len(rms_curve))
        metrics["silence_ratio"] = silence_ratio
        
        if silence_ratio > 0.5:
            warnings.append("High proportion of silence detected.")
    else:
        metrics["silence_ratio"] = 0.0
        
    # Pitch detection success / Voiced frame ratio
    if len(waveform) >= 2048:
        # librosa.pyin for pitch
        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sample_rate,
            frame_length=2048,
            hop_length=512,
            fill_na=None
        )
        voiced_ratio = float(np.mean(voiced_flag))
        metrics["voiced_frame_ratio"] = voiced_ratio
        metrics["pitch_detection_success_ratio"] = float(np.sum(~np.isnan(f0)) / len(f0))
        
        if voiced_ratio < 0.1:
            blocking_errors.append("Insufficient voiced content detected. Unable to estimate a stable vocal signal.")
        elif voiced_ratio < 0.3:
            warnings.append("Low voiced content ratio. Ensure continuous phonation.")
    else:
        metrics["voiced_frame_ratio"] = 0.0
        metrics["pitch_detection_success_ratio"] = 0.0
        
    # Zero crossing rate stats
    if len(waveform) > 0:
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=2048, hop_length=512)[0]
        metrics["zcr_mean"] = float(np.mean(zcr))
        metrics["zcr_std"] = float(np.std(zcr))
    else:
        metrics["zcr_mean"] = 0.0
        metrics["zcr_std"] = 0.0
        
    # Determine Status
    if len(blocking_errors) > 0:
        quality_status = "Poor"
        quality_score = 0.0
    elif len(warnings) > 1:
        quality_status = "Acceptable"
        quality_score = 0.5
    elif len(warnings) == 1:
        quality_status = "Acceptable"
        quality_score = 0.75
    else:
        quality_status = "Good"
        quality_score = 1.0
        
    return {
        "quality_status": quality_status,
        "quality_score": quality_score,
        "metrics": metrics,
        "warnings": warnings,
        "blocking_errors": blocking_errors
    }
