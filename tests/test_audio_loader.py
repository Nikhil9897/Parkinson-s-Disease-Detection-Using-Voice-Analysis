import numpy as np
import pytest
import io
import soundfile as sf
from src.audio.loader import load_audio, AudioLoadError, EmptyAudioError, UnsupportedAudioError
from config import AUDIO_SAMPLE_RATE

def create_dummy_wav(duration=1.0, sr=8000, channels=1):
    t = np.linspace(0, duration, int(sr * duration))
    # 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    if channels == 2:
        audio = np.column_stack((audio, audio))
        
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    buf.seek(0)
    return buf

def test_load_valid_audio():
    buf = create_dummy_wav(duration=1.0, sr=8000)
    waveform, sr, orig_sr, orig_ch, duration = load_audio(buf)
    
    assert sr == AUDIO_SAMPLE_RATE
    assert orig_sr == 8000
    assert orig_ch == 1
    assert waveform.ndim == 1
    assert waveform.dtype == np.float32
    assert abs(duration - 1.0) < 0.1

def test_load_stereo_audio():
    buf = create_dummy_wav(duration=1.0, sr=8000, channels=2)
    waveform, sr, orig_sr, orig_ch, duration = load_audio(buf)
    
    assert orig_ch == 2
    assert waveform.ndim == 1 # converted to mono

def test_unsupported_format():
    with pytest.raises(UnsupportedAudioError):
        load_audio("test.mp3")

def test_empty_audio():
    buf = create_dummy_wav(duration=0.0, sr=8000)
    with pytest.raises(EmptyAudioError):
        load_audio(buf)
