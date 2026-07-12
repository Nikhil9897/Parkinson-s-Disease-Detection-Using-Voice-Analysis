# Application Copy text

APP_TITLE = "NeuroVoice AI"
APP_SUBTITLE = "Temporal voice pattern analysis for Parkinsonian speech research"
DISCLAIMER_RESEARCH = "Research-use prototype. This system analyzes acoustic patterns and does not provide a medical diagnosis."

INSTRUCTIONS_TITLE = "Voice Test"
INSTRUCTIONS_DESC = "Hold the vowel “A” continuously for 5–10 seconds."
INSTRUCTIONS_LIST = [
    "Quiet room",
    "Natural volume",
    "15–20 cm microphone distance",
    "Continuous “AAAA” sound"
]

UPLOAD_TITLE = "Upload recording"
UPLOAD_HELP = "Support: WAV (Optionally MP3)"
BUTTON_ANALYZE = "Analyze Voice"

QUALITY_GOOD = "Good"
QUALITY_ACCEPT = "Acceptable"
QUALITY_POOR = "Poor"

BAND_LOWER = "Lower similarity"
BAND_INTERMEDIATE = "Intermediate similarity"
BAND_ELEVATED = "Elevated similarity"

INTERP_LOWER = "The analyzed recording shows lower similarity to Parkinsonian voice patterns learned by the model.\n\nThis output is intended for research and educational analysis. Voice patterns may be affected by age, vocal health, fatigue, recording conditions, microphone characteristics, and other factors."

INTERP_INTERMEDIATE = "The recording contains some acoustic characteristics that overlap with patterns learned from Parkinsonian speech samples.\n\nThis output is intended for research and educational analysis. Voice patterns may be affected by age, vocal health, fatigue, recording conditions, microphone characteristics, and other factors."

INTERP_ELEVATED = "The recording shows stronger similarity to Parkinsonian voice patterns represented in the training data.\n\nThis output is intended for research and educational analysis. Voice patterns may be affected by age, vocal health, fatigue, recording conditions, microphone characteristics, and other factors."

TEMPORAL_EXPLANATION = "Highlighted intervals contributed more strongly to the model output. Temporal importance does not establish clinical causality."
SHAP_EXPLANATION = "Model influence (not Disease cause or Clinical marker)."
