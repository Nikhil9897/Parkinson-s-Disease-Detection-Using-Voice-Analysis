import os
import json
import pandas as pd
from pathlib import Path

def build_metadata():
    project_root = Path("D:/NeuroVoice-AI")
    audit_csv = project_root / "reports" / "dataset_audio_audit.csv"
    excel_path = project_root / "Demographics_age_sex.xlsx"
    metadata_dir = project_root / "data" / "metadata"
    reports_dir = project_root / "reports"
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("Reading audit data and demographics...")
    audit_df = pd.read_csv(audit_csv)
    demographics_df = pd.read_excel(excel_path, engine="openpyxl")
    
    # Filename without extension usually maps to Sample ID
    audit_df["sample_id"] = audit_df["filename"].apply(lambda x: Path(x).stem)
    
    # Map label convention
    label_map = {"HC": 0, "PwPD": 1}
    class_map = {"HC": "Healthy Control", "PwPD": "Parkinson's Disease"}
    
    # Merge
    merged_df = pd.merge(
        audit_df, 
        demographics_df, 
        left_on="sample_id", 
        right_on="Sample ID", 
        how="outer", 
        indicator=True
    )
    
    successful_maps = merged_df[merged_df["_merge"] == "both"].copy()
    unmapped_audio = merged_df[merged_df["_merge"] == "left_only"].copy()
    unmapped_meta = merged_df[merged_df["_merge"] == "right_only"].copy()
    
    # Build final metadata
    final_records = []
    for _, row in successful_maps.iterrows():
        label_str = row["Label"]
        record = {
            "audio_path": row["audio_path"],
            "filename": row["filename"],
            "speaker_id": row["Sample ID"], # since 1 recording per speaker
            "label": label_map.get(label_str, 0),
            "class_name": class_map.get(label_str, "Healthy Control"),
            "age": row["Age"],
            "sex": row["Sex"],
            "recording_task": "sustained_vowel_a",
            "dataset_source": "figshare_pd_hc_voice_samples"
        }
        final_records.append(record)
        
    dataset_index = pd.DataFrame(final_records)
    dataset_index.to_csv(metadata_dir / "dataset_index.csv", index=False)
    
    if not unmapped_audio.empty:
        unmapped_audio.to_csv(reports_dir / "unmapped_samples.csv", index=False)
        
    report = {
        "total_audio_files": len(audit_df),
        "successfully_mapped_files": len(successful_maps),
        "unmapped_audio_files": len(unmapped_audio),
        "duplicate_identifiers": int(demographics_df["Sample ID"].duplicated().sum()),
        "metadata_rows_without_audio": len(unmapped_meta),
        "audio_files_without_metadata": len(unmapped_audio)
    }
    
    with open(reports_dir / "metadata_mapping_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Built metadata! {len(dataset_index)} successfully mapped files out of {len(audit_df)} audio files.")

if __name__ == "__main__":
    build_metadata()
