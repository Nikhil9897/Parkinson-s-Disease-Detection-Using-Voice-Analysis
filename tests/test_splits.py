"""
tests/test_splits.py

Mandatory speaker-overlap test per spec §11.
Verifies that for EVERY cross-validation fold, the set of training speakers
and validation speakers are completely disjoint.

Any intersection raises SpeakerLeakageError inside splits.py.
This test catches that before training.
"""

import pytest
import pandas as pd
import numpy as np
from src.training.splits import get_cross_validation_splits, SpeakerLeakageError
from config import N_SPLITS


def _make_df(n_hc: int = 20, n_pd: int = 20, one_per_speaker: bool = True):
    """
    Construct a synthetic dataset_index-style DataFrame.

    Parameters
    ----------
    n_hc : int  Number of healthy control samples.
    n_pd : int  Number of Parkinson's disease samples.
    one_per_speaker : bool
        If True each row has a unique speaker_id (independent participants).
        If False, duplicate speaker IDs are introduced to test group-KFold path.
    """
    total = n_hc + n_pd
    labels = [0] * n_hc + [1] * n_pd

    if one_per_speaker:
        speaker_ids = [f"SPK_{i:04d}" for i in range(total)]
    else:
        # 10 speakers, each with multiple recordings
        speaker_ids = [f"SPK_{i % 10:04d}" for i in range(total)]

    return pd.DataFrame({
        "audio_path": [f"fake_{i}.wav" for i in range(total)],
        "filename": [f"fake_{i}.wav" for i in range(total)],
        "speaker_id": speaker_ids,
        "label": labels,
        "class_name": ["Healthy Control"] * n_hc + ["Parkinson's Disease"] * n_pd,
    })


class TestSpeakerOverlapZero:
    """Core requirement: zero speaker overlap across all folds."""

    def test_no_overlap_one_per_speaker(self):
        """With independent participants, StratifiedKFold must produce no overlap."""
        df = _make_df(n_hc=20, n_pd=20, one_per_speaker=True)
        speaker_ids = df["speaker_id"].values

        for train_idx, val_idx in get_cross_validation_splits(df):
            train_speakers = set(speaker_ids[train_idx])
            val_speakers   = set(speaker_ids[val_idx])
            intersection   = train_speakers & val_speakers

            assert len(intersection) == 0, (
                f"Speaker leakage detected! Overlap: {intersection}"
            )

    def test_no_overlap_repeated_speakers(self):
        """With repeated speakers, StratifiedGroupKFold must produce no overlap."""
        df = _make_df(n_hc=25, n_pd=25, one_per_speaker=False)
        speaker_ids = df["speaker_id"].values

        for train_idx, val_idx in get_cross_validation_splits(df):
            train_speakers = set(speaker_ids[train_idx])
            val_speakers   = set(speaker_ids[val_idx])
            intersection   = train_speakers & val_speakers

            assert len(intersection) == 0, (
                f"Speaker leakage detected! Overlap: {intersection}"
            )

    def test_correct_number_of_folds(self):
        """Verify the generator yields exactly N_SPLITS folds."""
        df = _make_df(n_hc=20, n_pd=20, one_per_speaker=True)
        folds = list(get_cross_validation_splits(df))
        assert len(folds) == N_SPLITS

    def test_fold_indices_cover_all_samples(self):
        """All sample indices must appear in exactly one validation fold."""
        df = _make_df(n_hc=20, n_pd=20, one_per_speaker=True)
        total = len(df)
        seen_val = []

        for _, val_idx in get_cross_validation_splits(df):
            seen_val.extend(val_idx.tolist())

        assert len(seen_val) == total, (
            f"Expected {total} validation samples across folds, got {len(seen_val)}"
        )
        assert len(set(seen_val)) == total, (
            "Some samples appeared in validation more than once."
        )

    def test_stratification_preserves_class_balance(self):
        """Each validation fold should contain both classes."""
        df = _make_df(n_hc=20, n_pd=20, one_per_speaker=True)
        labels = df["label"].values

        for train_idx, val_idx in get_cross_validation_splits(df):
            val_labels = labels[val_idx]
            assert 0 in val_labels, "Validation fold missing class 0 (Healthy Control)"
            assert 1 in val_labels, "Validation fold missing class 1 (Parkinson's)"

    def test_leakage_error_type(self):
        """SpeakerLeakageError must be a subclass of Exception."""
        assert issubclass(SpeakerLeakageError, Exception)
