import os
import json
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import DATA_DIR, CACHE_DIR, MODELS_DIR
from src.audio.loader import load_audio
from src.audio.preprocessing import preprocess_audio
from src.audio.acoustic_features import extract_acoustic_features, get_acoustic_feature_schema

def get_file_hash(filepath: str) -> str:
    """Return MD5 hash of a file's content or modified time for speed."""
    # For speed on large datasets, using modified time + size is often enough, 
    # but we'll use MD5 of the first 1MB as a fingerprint.
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(1024 * 1024)
            hasher.update(buf)
            # Mix in the file size and mtime
            stat = os.stat(filepath)
            hasher.update(str(stat.st_size).encode())
            hasher.update(str(stat.st_mtime).encode())
        return hasher.hexdigest()
    except:
        return ""

def extract_features():
    print("Starting acoustic feature extraction...")
    
    metadata_path = DATA_DIR / "metadata" / "dataset_index.csv"
    if not metadata_path.exists():
        print("Metadata not found. Please run build_metadata.py first.")
        return
        
    df = pd.read_csv(metadata_path)
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_path = CACHE_DIR / "acoustic_features.csv" # using csv as pyarrow might not be installed
    
    # Load cache if exists
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
        cache_dict = cache_df.set_index('audio_path').to_dict('index')
    else:
        cache_dict = {}
        
    # Get schema
    schema = get_acoustic_feature_schema()
    with open(MODELS_DIR / "acoustic_feature_schema.json", "w") as f:
        json.dump(schema, f, indent=4)
        
    all_features = []
    updated = False
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        filepath = row["audio_path"]
        
        if not os.path.exists(filepath):
            continue
            
        file_hash = get_file_hash(filepath)
        
        # Check cache
        if filepath in cache_dict and cache_dict[filepath].get("file_hash") == file_hash:
            # Valid cache
            all_features.append(cache_dict[filepath])
            continue
            
        # Extract
        try:
            waveform, sr, _, _, _ = load_audio(filepath)
            processed_waveform, _ = preprocess_audio(waveform, sr)
            features = extract_acoustic_features(processed_waveform, sr)
            
            features["audio_path"] = filepath
            features["file_hash"] = file_hash
            all_features.append(features)
            updated = True
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    if updated or not cache_path.exists():
        final_df = pd.DataFrame(all_features)
        # Reorder columns: audio_path, file_hash, then schema
        cols = ["audio_path", "file_hash"] + [c for c in schema if c in final_df.columns]
        final_df = final_df[cols]
        final_df.to_csv(cache_path, index=False)
        print(f"\nFeature extraction complete. Cache saved to {cache_path}")
    else:
        print("\nAll files were cached. No updates needed.")

if __name__ == "__main__":
    extract_features()
