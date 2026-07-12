import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import DATA_DIR, CACHE_DIR
from src.audio.loader import load_audio
from src.audio.preprocessing import preprocess_audio
from src.audio.temporal_features import extract_temporal_features

def cache_temporal_features():
    print("Caching temporal features...")
    metadata_path = DATA_DIR / "metadata" / "dataset_index.csv"
    if not metadata_path.exists():
        print("Metadata not found.")
        return
        
    df = pd.read_csv(metadata_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    temporal_dict = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting temporal features"):
        filepath = row["audio_path"]
        
        if not os.path.exists(filepath):
            continue
            
        try:
            waveform, sr, _, _, _ = load_audio(filepath)
            processed_waveform, _ = preprocess_audio(waveform, sr)
            mfccs = extract_temporal_features(processed_waveform, sr)
            temporal_dict[filepath] = mfccs
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    np.savez_compressed(CACHE_DIR / "temporal_features.npz", **temporal_dict)
    print(f"Temporal features cached to {CACHE_DIR / 'temporal_features.npz'}")

if __name__ == "__main__":
    cache_temporal_features()
