import os
import glob
import json
import numpy as np
import pandas as pd
import soundfile as sf
import warnings
from pathlib import Path

def normalize_column_name(col_name):
    """Normalize column names for easier matching."""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    return col_name.strip().lower().replace(" ", "_").replace("-", "_")

def identify_columns(columns):
    """Attempt to identify semantic columns based on normalized names."""
    norm_cols = {col: normalize_column_name(col) for col in columns}
    
    identified = {
        "id_column": None,
        "group_column": None,
        "age_column": None,
        "sex_column": None
    }
    
    id_candidates = ["id", "sample_id", "subject_id", "participant_id", "speaker_id"]
    group_candidates = ["group", "class", "diagnosis", "status"]
    age_candidates = ["age"]
    sex_candidates = ["sex", "gender"]
    
    for orig_col, norm_col in norm_cols.items():
        if not identified["id_column"] and any(c == norm_col for c in id_candidates):
            identified["id_column"] = orig_col
        if not identified["group_column"] and any(c == norm_col for c in group_candidates):
            identified["group_column"] = orig_col
        if not identified["age_column"] and any(c == norm_col for c in age_candidates):
            identified["age_column"] = orig_col
        if not identified["sex_column"] and any(c == norm_col for c in sex_candidates):
            identified["sex_column"] = orig_col
            
    return identified

def audit_dataset():
    project_root = Path("D:/NeuroVoice-AI")
    hc_dir = project_root / "HC_AH"
    pd_dir = project_root / "PD_AH"
    excel_path = project_root / "Demographics_age_sex.xlsx"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("1. AUDIO FILE DISCOVERY")
    print("=" * 50)
    
    hc_files = []
    for root, dirs, files in os.walk(hc_dir):
        for file in files:
            hc_files.append(Path(root) / file)
            
    pd_files = []
    for root, dirs, files in os.walk(pd_dir):
        for file in files:
            pd_files.append(Path(root) / file)
            
    hc_wavs = [f for f in hc_files if f.suffix.lower() == '.wav']
    pd_wavs = [f for f in pd_files if f.suffix.lower() == '.wav']
    
    hc_other = [f for f in hc_files if f.suffix.lower() != '.wav']
    pd_other = [f for f in pd_files if f.suffix.lower() != '.wav']
    
    nested_hc = any(root != str(hc_dir) for root, dirs, files in os.walk(hc_dir))
    nested_pd = any(root != str(pd_dir) for root, dirs, files in os.walk(pd_dir))
    
    print(f"Total healthy-control recordings (WAV): {len(hc_wavs)}")
    print(f"Total Parkinson's recordings (WAV): {len(pd_wavs)}")
    print(f"Total recordings: {len(hc_wavs) + len(pd_wavs)}")
    
    print(f"\nOther files found in HC_AH: {len(hc_other)}")
    print(f"Other files found in PD_AH: {len(pd_other)}")
    
    print(f"\nNested directories in HC_AH: {nested_hc}")
    print(f"Nested directories in PD_AH: {nested_pd}")
    
    print("\nFirst 10 HC filenames:")
    for f in hc_wavs[:10]:
        print(f"  {f.name}")
        
    print("\nFirst 10 PD filenames:")
    for f in pd_wavs[:10]:
        print(f"  {f.name}")

    print("\n" + "=" * 50)
    print("2. EXCEL INSPECTION")
    print("=" * 50)
    
    try:
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names
        print(f"Sheet names: {sheet_names}")
        
        df = pd.read_excel(excel_path, engine='openpyxl')
        row_count, col_count = df.shape
        print(f"Number of rows: {row_count}")
        print(f"Number of columns: {col_count}")
        print(f"Exact column names: {list(df.columns)}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing-value count for every column:\n{df.isnull().sum()}")
        print(f"\nFirst 10 rows:\n{df.head(10).to_string()}")
        
        print("\nUnique values for categorical columns (dtype object):")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            print(f"  {col}: {df[col].nunique()} unique values")
            if df[col].nunique() < 20:
                print(f"    Values: {df[col].unique()}")
                
        identified_cols = identify_columns(df.columns)
        print(f"\nIdentified Columns:")
        for k, v in identified_cols.items():
            print(f"  {k}: {v}")
            
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        sheet_names = []
        df = pd.DataFrame()
        row_count, col_count = 0, 0

    print("\n" + "=" * 50)
    print("3. AUDIO METADATA AUDIT")
    print("=" * 50)
    
    audio_records = []
    durations = []
    sample_rates = {}
    channels_dist = {}
    unreadable = 0
    
    all_wavs = [(f, "HC_AH", 0) for f in hc_wavs] + [(f, "PD_AH", 1) for f in pd_wavs]
    
    print(f"Auditing {len(all_wavs)} audio files...")
    
    for filepath, source_folder, directory_label in all_wavs:
        record = {
            "audio_path": str(filepath.absolute()),
            "filename": filepath.name,
            "source_folder": source_folder,
            "directory_label": directory_label,
            "sample_rate": None,
            "channels": None,
            "duration_seconds": None,
            "num_samples": None,
            "readable": False,
            "error": None
        }
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = sf.info(filepath)
                
            record["sample_rate"] = info.samplerate
            record["channels"] = info.channels
            record["duration_seconds"] = info.duration
            record["num_samples"] = info.frames
            record["readable"] = True
            
            durations.append(info.duration)
            sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
            channels_dist[info.channels] = channels_dist.get(info.channels, 0) + 1
            
        except Exception as e:
            record["error"] = str(e)
            unreadable += 1
            
        audio_records.append(record)
        
    audit_df = pd.DataFrame(audio_records)
    audit_df.to_csv(reports_dir / "dataset_audio_audit.csv", index=False)
    print(f"Saved reports/dataset_audio_audit.csv")
    
    print("\n" + "=" * 50)
    print("4. DATASET SUMMARY")
    print("=" * 50)
    
    summary = {
        "healthy_recording_count": len(hc_wavs),
        "parkinsons_recording_count": len(pd_wavs),
        "total_recording_count": len(hc_wavs) + len(pd_wavs),
        "minimum_duration": float(np.min(durations)) if durations else None,
        "maximum_duration": float(np.max(durations)) if durations else None,
        "mean_duration": float(np.mean(durations)) if durations else None,
        "median_duration": float(np.median(durations)) if durations else None,
        "sample_rate_distribution": sample_rates,
        "channel_distribution": channels_dist,
        "unreadable_file_count": unreadable,
        "excel_sheet_names": sheet_names,
        "excel_columns": list(df.columns) if not df.empty else [],
        "excel_row_count": int(row_count)
    }
    
    with open(reports_dir / "dataset_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Saved reports/dataset_summary.json")
    print("\nAudit complete.")

if __name__ == "__main__":
    audit_dataset()
