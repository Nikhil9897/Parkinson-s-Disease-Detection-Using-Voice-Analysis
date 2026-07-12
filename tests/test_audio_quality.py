import numpy as np
from src.audio.quality import assess_audio_quality

def test_quality_good():
    # 3 second sine wave at 440 Hz
    t = np.linspace(0, 3.0, 3 * 16000)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    result = assess_audio_quality(waveform, sample_rate=16000)
    
    assert "quality_status" in result
    assert result["quality_status"] == "Good"
    assert len(result["blocking_errors"]) == 0

def test_quality_too_short():
    t = np.linspace(0, 0.5, int(0.5 * 16000))
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    result = assess_audio_quality(waveform, sample_rate=16000)
    assert len(result["blocking_errors"]) > 0
    assert any("too short" in e for e in result["blocking_errors"])

def test_quality_silent():
    waveform = np.zeros(3 * 16000, dtype=np.float32)
    result = assess_audio_quality(waveform, sample_rate=16000)
    
    assert len(result["blocking_errors"]) > 0
    assert any("silent" in e for e in result["blocking_errors"])
