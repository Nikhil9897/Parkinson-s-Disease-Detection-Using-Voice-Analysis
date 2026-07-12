from pathlib import Path

# Base Paths
PROJECT_ROOT = Path("D:/NeuroVoice-AI")

HC_AUDIO_DIR = PROJECT_ROOT / "HC_AH"
PD_AUDIO_DIR = PROJECT_ROOT / "PD_AH"
DEMOGRAPHICS_PATH = PROJECT_ROOT / "Demographics_age_sex.xlsx"

DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

# Audio Settings
AUDIO_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 1.0  # seconds
MAX_AUDIO_DURATION = 10.0 # seconds

# Feature Extraction Settings
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048
MAX_TIME_FRAMES = 431 # corresponds roughly to 10-15s audio with hop=512

# Training and Validation Settings
RANDOM_SEED = 42
N_SPLITS = 5
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.70
THRESHOLD_STEP = 0.01
