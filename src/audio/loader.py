import numpy as np
import librosa
import soundfile as sf
import io
import traceback
from typing import Tuple, Union
import sys

from config import AUDIO_SAMPLE_RATE

class AudioLoadError(Exception):
    """Exception raised for general audio loading failures."""
    pass

class EmptyAudioError(Exception):
    """Exception raised when the loaded audio contains no samples."""
    pass

class UnsupportedAudioError(Exception):
    """Exception raised when the audio format is not supported."""
    pass

def load_audio(file_path_or_bytes: Union[str, io.BytesIO]) -> Tuple[np.ndarray, int, int, int, float]:
    """
    Safely load an audio file.
    
    Args:
        file_path_or_bytes: Path to the audio file or file-like object.
        
    Returns:
        waveform (np.ndarray): Mono, float32, resampled waveform
        sample_rate (int): Target sample rate (e.g. 16000)
        original_sample_rate (int): Original sample rate
        original_channels (int): Original number of channels
        duration (float): Duration in seconds
        
    Raises:
        AudioLoadError: If audio cannot be loaded.
        EmptyAudioError: If audio is empty.
        UnsupportedAudioError: If format is unsupported.
    """
    try:
        if isinstance(file_path_or_bytes, str):
            if not file_path_or_bytes.lower().endswith('.wav'):
                raise UnsupportedAudioError("Only WAV format is supported.")
            
        with sf.SoundFile(file_path_or_bytes) as sound_file:
            original_sample_rate = sound_file.samplerate
            original_channels = sound_file.channels
            frames = sound_file.frames
            
            # Read all audio data
            waveform = sound_file.read(dtype='float32', always_2d=False)
            
            if frames == 0 or len(waveform) == 0:
                raise EmptyAudioError("The uploaded audio file is empty.")
                
            # Convert to mono if multi-channel
            if original_channels > 1:
                if waveform.ndim > 1:
                    waveform = librosa.to_mono(waveform.T)
                else:
                    # In case sf returned flattened
                    waveform = librosa.to_mono(waveform.reshape(-1, original_channels).T)
            
            # Verify finite values
            if not np.isfinite(waveform).all():
                raise AudioLoadError("Audio signal contains invalid (NaN/Inf) values.")
                
            # Resample
            if original_sample_rate != AUDIO_SAMPLE_RATE:
                waveform = librosa.resample(
                    y=waveform,
                    orig_sr=original_sample_rate,
                    target_sr=AUDIO_SAMPLE_RATE
                )
                
            duration = len(waveform) / AUDIO_SAMPLE_RATE
            
            return waveform, AUDIO_SAMPLE_RATE, original_sample_rate, original_channels, duration
            
    except UnsupportedAudioError:
        raise
    except EmptyAudioError:
        raise
    except AudioLoadError:
        raise
    except Exception as e:
        # We do not want to expose raw librosa/soundfile stack traces
        raise AudioLoadError(f"Failed to load audio: {str(e)}")
