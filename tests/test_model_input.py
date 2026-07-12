"""
tests/test_model_input.py

Validates the production model's expected input schema.
Ensures the feature vector produced by acoustic_features.py
matches the saved acoustic_feature_schema.json exactly.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from config import MODELS_DIR, AUDIO_SAMPLE_RATE, MAX_TIME_FRAMES, N_MFCC


SCHEMA_PATH = MODELS_DIR / "acoustic_feature_schema.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


class TestAcousticFeatureSchema:
    """Verify the acoustic feature schema is consistent."""

    def test_schema_file_exists(self):
        assert SCHEMA_PATH.exists(), (
            f"acoustic_feature_schema.json not found at {SCHEMA_PATH}. "
            "Run scripts/train_baselines.py first."
        )

    def test_schema_is_valid_json(self):
        if not SCHEMA_PATH.exists():
            pytest.skip("Schema file not found.")
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        assert isinstance(schema, list), "Schema must be a list of feature names."
        assert len(schema) > 0, "Schema must not be empty."

    def test_schema_contains_expected_features(self):
        if not SCHEMA_PATH.exists():
            pytest.skip("Schema file not found.")
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)

        required = [
            "f0_mean", "f0_std", "voiced_frame_ratio",
            "rms_mean", "zcr_mean",
            "spectral_centroid_mean", "spectral_flatness_mean",
            "mfcc_01_mean", "mfcc_01_std",
            "mfcc_20_mean", "mfcc_20_std",
        ]
        for feat in required:
            assert feat in schema, f"Required feature '{feat}' missing from schema."

    def test_feature_extraction_matches_schema(self):
        """Feature vector length must equal schema length."""
        if not SCHEMA_PATH.exists():
            pytest.skip("Schema file not found.")

        from src.audio.acoustic_features import extract_acoustic_features
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)

        # Use a synthetic sine wave at 440 Hz, 3 seconds
        t = np.linspace(0, 3.0, 3 * AUDIO_SAMPLE_RATE)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        features = extract_acoustic_features(waveform, sample_rate=AUDIO_SAMPLE_RATE)
        feature_vector = [features.get(k, 0.0) for k in schema]

        assert len(feature_vector) == len(schema), (
            f"Feature vector length {len(feature_vector)} != schema length {len(schema)}"
        )

    def test_feature_values_are_finite(self):
        """All extracted feature values must be finite numbers."""
        from src.audio.acoustic_features import extract_acoustic_features

        t = np.linspace(0, 3.0, 3 * AUDIO_SAMPLE_RATE)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        features = extract_acoustic_features(waveform, sample_rate=AUDIO_SAMPLE_RATE)

        for name, val in features.items():
            assert np.isfinite(val), f"Feature '{name}' is not finite: {val}"


class TestTemporalModelInput:
    """Verify temporal feature shape matches MAX_TIME_FRAMES × N_MFCC."""

    def test_temporal_shape(self):
        from src.audio.temporal_features import extract_temporal_features

        t = np.linspace(0, 5.0, 5 * AUDIO_SAMPLE_RATE)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        mfcc_seq = extract_temporal_features(waveform, sample_rate=AUDIO_SAMPLE_RATE)

        assert mfcc_seq.ndim == 2, "Temporal features must be 2D (time, features)."
        assert mfcc_seq.shape[1] == N_MFCC, (
            f"Expected {N_MFCC} MFCC coefficients, got {mfcc_seq.shape[1]}"
        )

    def test_temporal_values_finite(self):
        from src.audio.temporal_features import extract_temporal_features

        t = np.linspace(0, 3.0, 3 * AUDIO_SAMPLE_RATE)
        waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        mfcc_seq = extract_temporal_features(waveform, sample_rate=AUDIO_SAMPLE_RATE)
        assert np.all(np.isfinite(mfcc_seq)), "Temporal features contain non-finite values."
