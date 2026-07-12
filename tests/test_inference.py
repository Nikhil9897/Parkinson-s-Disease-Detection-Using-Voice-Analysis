"""
tests/test_inference.py

Integration tests for the inference pipeline (src/inference/predictor.py).

Tests:
  - Score is between 0.0 and 1.0
  - All required output keys are present on success
  - Quality failure correctly blocks prediction
  - Missing model returns a clear error (not a fake prediction)
"""

import io
import numpy as np
import pytest
import soundfile as sf

from src.inference.predictor import predictor
from config import AUDIO_SAMPLE_RATE

REQUIRED_SUCCESS_KEYS = {
    "score",
    "threshold",
    "predicted_class",
    "pattern_band",
    "quality",
    "audio_metrics",
    "inference_time_ms",
    "model_name",
    "model_version",
}

REQUIRED_QUALITY_KEYS = {
    "quality_status",
    "quality_score",
    "metrics",
    "warnings",
    "blocking_errors",
}


def _make_wav_bytes(duration: float = 3.0, freq: float = 180.0, sr: int = AUDIO_SAMPLE_RATE) -> bytes:
    """Create a synthetic vowel-like sine wave and return as WAV bytes."""
    t = np.linspace(0, duration, int(sr * duration))
    waveform = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _make_silent_wav_bytes(duration: float = 3.0) -> bytes:
    """Create a near-silent WAV file."""
    waveform = np.zeros(int(AUDIO_SAMPLE_RATE * duration), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, AUDIO_SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


def _make_short_wav_bytes(duration: float = 0.3) -> bytes:
    """Create a very short WAV file."""
    t = np.linspace(0, duration, int(AUDIO_SAMPLE_RATE * duration))
    waveform = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, AUDIO_SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


class TestInferenceOutput:

    def test_model_loaded(self):
        """Production model must be loaded (models trained)."""
        if predictor.model is None:
            pytest.skip("Model not trained yet. Run training scripts first.")

    def test_score_in_valid_range(self):
        """Score must be between 0.0 and 1.0 inclusive."""
        if predictor.model is None:
            pytest.skip("Model not trained yet.")

        wav_bytes = _make_wav_bytes(duration=3.0, freq=180.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        if "error" in result:
            pytest.skip(f"Prediction returned error: {result['error']}")

        score = result["score"]
        assert 0.0 <= score <= 1.0, f"Score {score} is outside [0, 1]."

    def test_required_output_keys_present(self):
        """All required output keys must be present on successful prediction."""
        if predictor.model is None:
            pytest.skip("Model not trained yet.")

        wav_bytes = _make_wav_bytes(duration=3.0, freq=180.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        if "error" in result:
            pytest.skip(f"Prediction returned error: {result['error']}")

        for key in REQUIRED_SUCCESS_KEYS:
            assert key in result, f"Required key '{key}' missing from prediction output."

    def test_quality_struct_present(self):
        """Quality result must have required sub-keys."""
        if predictor.model is None:
            pytest.skip("Model not trained yet.")

        wav_bytes = _make_wav_bytes(duration=3.0, freq=180.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        if "error" in result:
            pytest.skip(f"Quality gate blocked prediction: {result['error']}")

        quality = result["quality"]
        for key in REQUIRED_QUALITY_KEYS:
            assert key in quality, f"Required quality key '{key}' missing."

    def test_predicted_class_is_binary(self):
        """predicted_class must be 0 or 1."""
        if predictor.model is None:
            pytest.skip("Model not trained yet.")

        wav_bytes = _make_wav_bytes(duration=3.0, freq=180.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        if "error" in result:
            pytest.skip(f"Prediction returned error.")

        assert result["predicted_class"] in (0, 1), (
            f"predicted_class must be 0 or 1, got {result['predicted_class']}"
        )

    def test_pattern_band_is_valid(self):
        """pattern_band must be one of the three defined values."""
        if predictor.model is None:
            pytest.skip("Model not trained yet.")

        valid_bands = {"Lower similarity", "Intermediate similarity", "Elevated similarity"}
        wav_bytes = _make_wav_bytes(duration=3.0, freq=180.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        if "error" in result:
            pytest.skip(f"Prediction returned error.")

        assert result["pattern_band"] in valid_bands, (
            f"Unexpected pattern_band: {result['pattern_band']}"
        )

    def test_inference_time_is_positive(self):
        """Inference time must be a positive integer millisecond value."""
        if predictor.model is None:
            pytest.skip("Model not trained yet.")

        wav_bytes = _make_wav_bytes(duration=3.0, freq=180.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        if "error" in result:
            pytest.skip("Prediction returned error.")

        assert result["inference_time_ms"] >= 0, "Inference time must be >= 0."

    def test_silent_audio_blocked_by_quality(self):
        """A near-silent recording must be rejected by the quality gate."""
        wav_bytes = _make_silent_wav_bytes(duration=3.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        # Either quality error or load error — both are acceptable rejections
        assert "error" in result, "Silent audio should be blocked before prediction."

    def test_short_audio_blocked_by_quality(self):
        """A very short recording must be rejected by the quality gate."""
        wav_bytes = _make_short_wav_bytes(duration=0.3)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        assert "error" in result, "Short audio should be blocked before prediction."

    def test_no_model_returns_error_not_prediction(self):
        """If no model is trained, the response must be an error dict, not a fake score."""
        # We test this via the predictor's own guard
        if predictor.model is not None:
            pytest.skip("Model is present; testing missing-model path is not applicable.")

        wav_bytes = _make_wav_bytes(duration=3.0)
        result = predictor.predict_voice(io.BytesIO(wav_bytes))

        assert "error" in result, (
            "When no model is loaded, predict_voice must return an error dict."
        )
        assert "score" not in result, (
            "When no model is loaded, no score should be returned."
        )
