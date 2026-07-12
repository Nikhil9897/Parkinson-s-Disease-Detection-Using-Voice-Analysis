# NeuroVoice AI

**Explainable Temporal Voice Pattern Screening for Parkinsonian Speech**

A research and educational prototype that analyzes sustained vowel voice recordings for acoustic patterns statistically associated with Parkinsonian speech. Built entirely as an independent implementation — not derived from any third-party Parkinson's detection repository.

---

## Project Overview

NeuroVoice AI processes a prolonged "AAAA" vowel recording through a full signal processing and machine learning pipeline. It extracts both classical acoustic features (F0, energy, spectral, MFCC) and temporal MFCC sequences for deep learning, evaluates all models with speaker-disjoint cross-validation, and deploys the best-performing model behind a Streamlit research interface.

---

## Research Objective

To demonstrate a defensible, end-to-end ML pipeline for voice-based Parkinsonian speech pattern analysis — with correct leakage-aware evaluation, threshold calibration, model comparison, and explainability — in a responsible, clearly non-diagnostic interface.

---

## Why Voice Analysis

Parkinson's disease affects the motor control of phonation. Dysarthria and hypokinetic dysphonia in PD manifest as measurable changes in pitch stability, energy, spectral structure, and temporal regularity. Sustained vowel phonation ("AAAA") isolates these phonatory characteristics from articulatory interference, making it a common protocol in voice-based PD research.

---

## System Architecture

```
Audio File
    │
    ▼
src/audio/loader.py         — Safe loading, mono conversion, resampling to 16 kHz
    │
    ▼
src/audio/preprocessing.py  — DC offset removal, silence trim, amplitude normalization
    │
    ▼
src/audio/quality.py        — Duration, RMS, silence ratio, clipping, voiced frame ratio
    │
    ├──► src/audio/acoustic_features.py  — Classical tabular features (F0, energy, spectral, MFCC×20)
    │        │
    │        ▼
    │    src/models/svm_model.py          — RobustScaler + SVC (RBF)
    │    src/models/gradient_boosting.py  — HistGradientBoostingClassifier
    │
    └──► src/audio/temporal_features.py  — 40 MFCC × time frames (padded/truncated)
             │
             ▼
         src/models/temporal_model.py    — Conv1D → BiLSTM → TemporalAttention → Dense
         src/models/attention.py         — Custom Keras temporal attention layer
             │
             ▼
         src/training/splits.py          — StratifiedKFold / StratifiedGroupKFold (leakage-proof)
         src/training/threshold.py       — OOF threshold optimization
         src/training/metrics.py         — Full metric set with safe zero-division
             │
             ▼
         scripts/evaluate_models.py      — Model comparison, production selection
             │
             ▼
         models/model_metadata.json      — Production artifact registry
             │
             ▼
         src/inference/predictor.py      — Unified prediction API
         src/inference/model_loader.py   — @st.cache_resource loader
             │
             ▼
         app.py                          — Streamlit research interface
```

---

## Dataset

**Title:** PD and HC Voice Samples — Sustained Vowel "AAAA"

**Source:** Figshare — [figshare.com](https://figshare.com)

**Content:**
- 81 total recordings
- 41 Healthy Control (HC_AH/)
- 40 Parkinson's Disease (PD_AH/)
- Sustained vowel /a/ phonation

**Expected recording protocol:** Sustained "AAAA" for 5–10 seconds.

---

## Dataset License and Attribution

This dataset is sourced from Figshare under its original license. The NeuroVoice AI code is an **independent implementation** and is not affiliated with the dataset authors.

All dataset files remain in their original form. No audio files were modified.

The code in this repository — including all model architecture, training logic, feature extraction, inference pipeline, and application design — is an original implementation.

**Dataset citation:** Please cite the original Figshare dataset record when using this data in academic work.

---

## Expected Voice Protocol

Participants were asked to sustain the vowel "AAAA" for several seconds at a natural volume and comfortable pitch, recorded in a quiet environment.

For the Streamlit application, uploaded recordings should follow the same protocol for meaningful pattern comparison.

---

## Data Audit

Run the dataset audit before any training:

```bash
python scripts/audit_dataset.py
```

Outputs:
- `reports/dataset_audio_audit.csv` — per-file metadata
- `reports/dataset_summary.json` — aggregate statistics

---

## Audio Processing Pipeline

1. **Loading** (`src/audio/loader.py`): Accepts WAV files or BytesIO streams. Converts multi-channel to mono. Validates finite values. Resamples to 16 kHz. Raises `AudioLoadError`, `EmptyAudioError`, `UnsupportedAudioError`.

2. **Preprocessing** (`src/audio/preprocessing.py`): DC offset subtraction → librosa silence trimming → conservative peak normalization. No noise suppression, no pitch modification, no time stretching.

3. **Quality Assessment** (`src/audio/quality.py`): Duration check, RMS energy threshold, silence ratio, clipping detection, voiced frame ratio via pyin. Produces `Good / Acceptable / Poor` rating with structured `blocking_errors`.

---

## Acoustic Features

Classical tabular features extracted per file:

| Group | Features |
|---|---|
| Pitch (F0) | mean, median, std, min, max, range, IQR, voiced_frame_ratio |
| Perturbation | estimated_local_jitter (approx. — not MDVP equivalent) |
| Energy | RMS mean, std, median, IQR |
| Temporal | ZCR mean, std |
| Spectral | centroid, bandwidth, rolloff, flatness (mean + std each) |
| MFCC | 20 coefficients × {mean, std} |
| Delta MFCC | 20 delta coefficients × {mean, std} |

**Total: 98 features.** Schema saved to `models/acoustic_feature_schema.json`.

---

## Temporal MFCC Representation

For the CNN-BiLSTM model:
- 40 MFCC coefficients extracted per frame
- Frames preserved in time order: shape = `(time_frames, 40)`
- Padded or truncated to `MAX_TIME_FRAMES` (config.py)
- Normalization statistics derived only from training data

---

## Model Architectures

### SVM
- `RobustScaler` → `SVC(kernel='rbf', probability=True)`
- Scikit-learn Pipeline
- Grid search over C, gamma with speaker-disjoint CV

### Gradient Boosting
- `HistGradientBoostingClassifier`
- Handles missing values natively
- Feature importance via SHAP (KernelExplainer)

### CNN-BiLSTM
```
Input (MAX_TIME_FRAMES × 40)
  → Conv1D(64, kernel=5, padding='same') → BatchNorm → GELU → MaxPool(2) → Dropout(0.15)
  → Conv1D(96, kernel=3, padding='same') → BatchNorm → GELU → MaxPool(2) → Dropout(0.20)
  → Bidirectional LSTM(64, return_sequences=True) → Dropout
  → TemporalAttention (custom Keras layer)
  → Dense(64, GELU) → Dropout(0.35)
  → Dense(16, GELU)
  → Dense(1, Sigmoid)
```

Parameter count is intentionally small given the dataset size.

---

## Why BiLSTM

A unidirectional LSTM can only use past context. For sustained phonation analysis, both preceding and following frames are informative (e.g., pitch transitions require bidirectional context). The BiLSTM reads the sequence in both directions and concatenates hidden states.

---

## Speaker Leakage Prevention

All cross-validation splits are checked for speaker-set intersection before training:

```python
# src/training/splits.py
if train_speakers ∩ val_speakers ≠ ∅:
    raise SpeakerLeakageError(...)
```

- If all speakers have exactly 1 recording: `StratifiedKFold` is used.
- If any speaker has multiple recordings: `StratifiedGroupKFold` is used.

Split strategy documented in `reports/split_strategy.json`.

---

## Cross-Validation Strategy

- 5-fold stratified cross-validation
- Speaker-disjoint (verified per fold)
- Out-of-fold predictions saved to `reports/oof_predictions.csv`
- Fold-level metrics in `reports/fold_metrics.csv`
- Training histories in `reports/training_history_fold_*.csv`

---

## Threshold Optimization

Decision threshold is optimized on out-of-fold predictions (not on any test set):

```
Sweep: 0.30 → 0.70 (step 0.01)
Objective: maximum F1-score
Tiebreak 1: highest recall
Tiebreak 2: closest to 0.50
```

Result saved to `models/model_metadata.json`.

---

## Model Comparison

All three models are evaluated on the same speaker-disjoint folds. The production model is selected by:

1. **Primary:** Mean F1-score (cross-validation)
2. **Secondary:** Mean ROC-AUC
3. **Inspection:** Standard deviation, Recall, Specificity

Full comparison in `reports/model_comparison.csv`.

---

## Results

Results are read from generated reports. Run the full pipeline to populate:

```bash
python scripts/audit_dataset.py
python scripts/build_metadata.py
python scripts/extract_features.py
python scripts/cache_temporal.py
python scripts/train_baselines.py
python scripts/train_temporal_model.py
python scripts/evaluate_models.py
```

Then inspect:
- `reports/model_comparison.csv`
- `reports/fold_metrics.csv`
- `models/model_metadata.json` → `validation_metrics`

---

## Streamlit Application

The application is for inference only. Training is handled in `scripts/`.

Features:
- Audio upload with immediate duration/sample-rate metadata display
- Staged analysis progress (5 real stages)
- Horizontal probability scale with threshold marker
- 4 analysis tabs: Overview, Signal Analysis, Model Insights, Research
- Mel Spectrogram and MFCC Heatmap (toggleable)
- SHAP feature influence chart (classical models)
- Temporal attention overlay (CNN-BiLSTM)
- Full research metadata in Research view
- `@st.cache_resource` for all model artifacts — loaded once per session

---

## Installation

### Prerequisites

- Python 3.10+

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Place the dataset in the project root:

```
D:\NeuroVoice-AI\
├── HC_AH\          ← healthy control WAV files
├── PD_AH\          ← Parkinson's WAV files
└── Demographics_age_sex.xlsx
```

---

## Dataset Audit Command

```bash
python scripts/audit_dataset.py
```

---

## Metadata Build Command

```bash
python scripts/build_metadata.py
```

---

## Feature Extraction Command

```bash
python scripts/extract_features.py
python scripts/cache_temporal.py
```

---

## Baseline Training Command

```bash
python scripts/train_baselines.py
```

---

## Temporal Model Training Command

```bash
python scripts/train_temporal_model.py
```

---

## Application Launch Command

```bash
$env:PYTHONPATH="D:\NeuroVoice-AI"
python -m streamlit run app.py
```

---

## Project Structure

```
D:\NeuroVoice-AI\
├── HC_AH\                          # Healthy control audio (original, unmodified)
├── PD_AH\                          # Parkinson's audio (original, unmodified)
├── Demographics_age_sex.xlsx       # Dataset demographics
├── app.py                          # Streamlit application (inference only)
├── config.py                       # Central configuration (all constants)
├── requirements.txt
├── README.md
├── data\
│   ├── metadata\dataset_index.csv  # Unified sample table with labels
│   └── cache\                      # Extracted feature caches
├── scripts\                        # Standalone runnable scripts
├── src\
│   ├── audio\                      # Loading, preprocessing, quality, features
│   ├── models\                     # SVM, GB, CNN-BiLSTM, Attention
│   ├── training\                   # Splits, metrics, threshold, callbacks
│   ├── explainability\             # SHAP, temporal saliency
│   ├── inference\                  # Predictor, model_loader
│   └── ui\                         # Styles, components, charts, copy
├── models\                         # Trained artifacts + metadata
├── reports\                        # Audit, fold metrics, comparisons
├── visualizations\
├── notebooks\
└── tests\                          # pytest test suite
```

---

## Limitations

- **Small dataset (81 samples):** Cross-validation metrics should be interpreted with caution. Confidence intervals are wide.
- **Single recording task:** Only sustained vowel /a/ is analyzed. Connected speech, reading tasks, and running speech are not covered.
- **No noise suppression:** Recordings with background noise may produce unreliable quality scores and predictions.
- **Approximate perturbation features:** `estimated_local_jitter` is a Librosa-derived approximation. It is not equivalent to validated clinical MDVP measurements.
- **Temporal attention ≠ clinical marker:** Highlighted intervals indicate model sensitivity, not pathological events.

---

## Ethical Considerations

This system:
- Is a **research and educational prototype only**
- Does **not** provide medical diagnoses
- Does **not** recommend clinical treatment
- Does **not** replace professional neurological evaluation
- Should **not** be used for clinical decision-making

Outputs use the terminology:
- *Parkinsonian voice pattern score* (not "Parkinson's probability")
- *Pattern similarity* (not "disease certainty")
- *Research prototype* (not "AI doctor")

---

## Medical Disclaimer

> **This application analyzes acoustic patterns for research and educational purposes only. It does not provide a medical diagnosis, clinical guidance, or therapeutic recommendation. Voice patterns are affected by age, vocal health, fatigue, recording environment, and microphone characteristics. Any health concerns should be evaluated by a qualified medical professional.**

---

## Future Work

- Extend to connected speech and reading passage tasks
- Incorporate validated MDVP-equivalent jitter/shimmer via Praat/Parselmouth
- Collect a larger, demographically diverse dataset
- Add multi-language support
- Longitudinal tracking (repeated recordings over time)
- Calibration curves for probability reliability

---

## Dataset Citation

Please cite the original Figshare dataset when using these audio recordings in academic work. Refer to the original Figshare record for the full attribution requirements.

**NeuroVoice AI code** is an original independent implementation and should be cited separately if referenced in academic work.
