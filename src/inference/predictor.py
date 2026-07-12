import io
import time
import json
import joblib
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Union

from config import MODELS_DIR
from src.audio.loader import load_audio
from src.audio.preprocessing import preprocess_audio
from src.audio.quality import assess_audio_quality
from src.audio.acoustic_features import extract_acoustic_features, get_acoustic_feature_schema
from src.audio.temporal_features import extract_temporal_features

class VoicePredictor:
    def __init__(self):
        self.metadata_path = MODELS_DIR / "model_metadata.json"
        self.metadata = None
        self.model = None
        self.schema = None
        self.norm_stats = None
        
        self._load_artifacts()
        
    def _load_artifacts(self):
        if not self.metadata_path.exists():
            return
            
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
            
        prod_model = self.metadata.get("production_model")
        
        if prod_model == "SVM":
            self.model = joblib.load(MODELS_DIR / "svm_pipeline.joblib")
            with open(MODELS_DIR / "acoustic_feature_schema.json", "r") as f:
                self.schema = json.load(f)
        elif prod_model == "Gradient Boosting":
            self.model = joblib.load(MODELS_DIR / "boosting_model.joblib")
            with open(MODELS_DIR / "acoustic_feature_schema.json", "r") as f:
                self.schema = json.load(f)
        elif prod_model == "CNN-BiLSTM":
            self.model = tf.keras.models.load_model(MODELS_DIR / "cnn_bilstm.keras", compile=False)
            norm_stats = np.load(MODELS_DIR / "temporal_norm_stats.npz")
            self.norm_stats = {"mean": norm_stats["mean"], "std": norm_stats["std"]}
            
    def predict_voice(self, audio_input: Union[str, io.BytesIO]) -> Dict[str, Any]:
        """Runs the unified prediction pipeline."""
        if self.metadata is None or self.model is None:
            return {"error": "Model not trained or artifacts missing. Please train models first."}
            
        start_time = time.time()
        
        # 1. Load Audio
        try:
            waveform, sr, orig_sr, orig_ch, duration = load_audio(audio_input)
        except Exception as e:
            return {"error": f"Audio loading failed: {str(e)}"}
            
        # 2. Preprocess
        processed_waveform, processing_meta = preprocess_audio(waveform, sr)
        
        # 3. Quality Assessment
        quality = assess_audio_quality(processed_waveform, sr)
        if len(quality["blocking_errors"]) > 0:
            return {
                "error": "Recording failed quality assessment.",
                "quality": quality
            }
            
        prod_model = self.metadata.get("production_model")
        
        # 4. Feature Extraction & Prediction
        try:
            if prod_model in ["SVM", "Gradient Boosting"]:
                features = extract_acoustic_features(processed_waveform, sr)
                # Order by schema
                feature_vector = np.array([[features.get(k, 0.0) for k in self.schema]])
                prob = self.model.predict_proba(feature_vector)[0, 1]
                
            elif prod_model == "CNN-BiLSTM":
                mfccs = extract_temporal_features(processed_waveform, sr)
                # Normalize
                mfccs_norm = (mfccs - self.norm_stats["mean"]) / self.norm_stats["std"]
                mfccs_batch = np.expand_dims(mfccs_norm, axis=0)
                
                prob = float(self.model.predict(mfccs_batch, verbose=0)[0, 0])
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}
            
        inference_time_ms = int((time.time() - start_time) * 1000)
        
        # 5. Threshold application
        threshold = self.metadata.get("selected_threshold", 0.5)
        
        if prob < threshold - 0.1:
            pattern_band = "Lower similarity"
        elif prob > threshold + 0.1:
            pattern_band = "Elevated similarity"
        else:
            pattern_band = "Intermediate similarity"
            
        predicted_class = 1 if prob >= threshold else 0
        
        # We can add SHAP here later if requested
        explanation = None
        
        return {
            "score": prob,
            "threshold": threshold,
            "predicted_class": predicted_class,
            "pattern_band": pattern_band,
            "quality": quality,
            "audio_metrics": processing_meta,
            "explanation": explanation,
            "inference_time_ms": inference_time_ms,
            "model_name": prod_model,
            "model_version": self.metadata.get("project_version")
        }

predictor = VoicePredictor()
