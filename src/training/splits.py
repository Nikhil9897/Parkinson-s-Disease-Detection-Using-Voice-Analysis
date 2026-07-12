import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from typing import Tuple, Generator

from config import N_SPLITS, RANDOM_SEED

class SpeakerLeakageError(Exception):
    pass

def get_cross_validation_splits(df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Data-driven cross-validation splitting.
    Verifies that train and validation sets have zero speaker overlap.
    """
    labels = df["label"].values
    speaker_ids = df["speaker_id"].values
    
    unique_speakers = df["speaker_id"].nunique()
    total_samples = len(df)
    
    if unique_speakers == total_samples:
        # One recording per independent speaker
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        splits = cv.split(df, labels)
    else:
        # Multiple recordings per speaker
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        splits = cv.split(df, labels, groups=speaker_ids)
        
    for fold, (train_idx, val_idx) in enumerate(splits):
        train_speakers = set(speaker_ids[train_idx])
        val_speakers = set(speaker_ids[val_idx])
        
        intersection = train_speakers.intersection(val_speakers)
        if len(intersection) > 0:
            raise SpeakerLeakageError(f"Fold {fold} has speaker leakage! Overlapping speakers: {intersection}")
            
        yield train_idx, val_idx
