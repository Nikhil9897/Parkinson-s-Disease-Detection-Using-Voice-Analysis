"""
app.py — NeuroVoice AI Streamlit Application

Inference and visualization client for Parkinsonian voice pattern research.
"""

import streamlit as st

st.set_page_config(
    page_title="NeuroVoice AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force hot-reload and IDE resolution update
import hashlib
import io
import json
import pandas as pd
import numpy as np

# UI & Styles
from src.ui.styles import get_css
from src.ui.components import (
    render_header, render_instructions, render_result_summary,
)
from src.ui.charts import (
    create_score_scale, create_waveform_chart, create_f0_chart,
    create_mel_spectrogram, create_mfcc_heatmap, create_shap_chart,
    create_attention_overlay,
)
from src.ui import copy
from src.inference.model_loader import (
    load_metadata, load_production_model, load_feature_schema, load_norm_stats,
    load_training_features,
)
from config import REPORTS_DIR, AUDIO_SAMPLE_RATE

# 25 MB Limit
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# State Management & Helpers
# ---------------------------------------------------------------------------

def _audio_hash(audio_bytes: bytes) -> str:
    return hashlib.md5(audio_bytes).hexdigest()


def _clear_analysis_state():
    """Clear all analysis-related session state when a new file is uploaded."""
    for key in ["prediction_result", "processed_waveform", "processed_sr",
                "acoustic_feats", "analysis_done"]:
        st.session_state[key] = None
    st.session_state["analysis_done"] = False


def _init_session_state():
    defaults = {
        "audio_hash": None,
        "audio_bytes": None,
        "prediction_result": None,
        "processed_waveform": None,
        "processed_sr": None,
        "acoustic_feats": None,
        "analysis_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Sidebar Renders
# ---------------------------------------------------------------------------

def _render_sidebar(model_type: str):
    with st.sidebar:
        st.markdown("### ANALYSIS")
        view_mode = st.radio(
            "Analysis view",
            ["Standard", "Detailed", "Research"],
            index=0,
            key="view_mode",
        )

        show_melspec = True
        show_mfcc = False
        show_temporal = True

        # Only show visualization options in Detailed or Research views
        if view_mode in ["Detailed", "Research"]:
            st.markdown("### VISUALIZATION")
            show_melspec = st.checkbox("Mel Spectrogram", value=True, key="show_melspec")
            show_mfcc = st.checkbox("MFCC Heatmap", value=False, key="show_mfcc")
            # Only show temporal attention checkbox if the production model is CNN-BiLSTM
            if model_type == "cnn_bilstm":
                show_temporal = st.checkbox("Temporal attention", value=True, key="show_temporal")
            else:
                show_temporal = False
        else:
            show_temporal = False

        st.markdown("### SYSTEM")
        metadata = load_metadata()
        if metadata:
            raw_model = metadata.get("production_model", "—")
            
            # Formatting raw values for scientific workstations
            input_rep = "Acoustic voice features"
            if raw_model == "CNN-BiLSTM":
                input_rep = "Temporal voice sequence"

            st.markdown(f"**Production model**  \n{raw_model}")
            st.markdown(f"**Input representation**  \n{input_rep}")
            st.markdown(f"**Voice protocol**  \nSustained vowel /a/")
            st.markdown(f"**Model version**  \nv1.0")
        else:
            st.warning("No trained model found.")

        st.markdown("### ABOUT")
        with st.expander("Workspace Details", expanded=False):
            st.markdown(
                "**Research Prototype**  \n"
                "Designed for Parkinsonian speech pattern analysis. "
                "Not clinically validated."
            )
            st.markdown(
                "**Medical Disclaimer**  \n"
                "Acoustic features can vary due to environmental noise, "
                "microphones, or age. Consult a neurologist for health concerns."
            )
            st.markdown(
                "**Dataset Attribution**  \n"
                "Phonation samples derived from Figshare PD/HC voice database."
            )

    return view_mode, show_melspec, show_mfcc, show_temporal


# ---------------------------------------------------------------------------
# Staged Analysis Pipeline
# ---------------------------------------------------------------------------

def _run_analysis(audio_bytes: bytes) -> dict:
    from src.audio.loader import load_audio
    from src.audio.preprocessing import preprocess_audio
    from src.audio.acoustic_features import extract_acoustic_features

    steps = [
        "Validating recording",
        "Processing signal",
        "Extracting acoustic features",
        "Running model inference",
        "Preparing analysis",
    ]
    
    # Render blue/neutral progress bar
    progress_bar = st.progress(0, text=steps[0])

    # Stage 1: Validate
    progress_bar.progress(0.15, text=steps[0])
    
    # Stage 2: Preprocess
    progress_bar.progress(0.35, text=steps[1])
    try:
        wav, sr, _, _, _ = load_audio(io.BytesIO(audio_bytes))
        proc_wav, proc_meta = preprocess_audio(wav, sr)
        st.session_state["processed_waveform"] = proc_wav
        st.session_state["processed_sr"] = sr
    except Exception:
        proc_wav = None

    # Stage 3: Feature extraction
    progress_bar.progress(0.60, text=steps[2])
    if proc_wav is not None:
        try:
            feats = extract_acoustic_features(proc_wav, sr)
            st.session_state["acoustic_feats"] = feats
        except Exception:
            pass

    # Stage 4: Inference
    progress_bar.progress(0.80, text=steps[3])
    from src.inference.predictor import predictor as _predictor
    result = _predictor.predict_voice(io.BytesIO(audio_bytes))

    # Stage 5: Done
    progress_bar.progress(1.0, text=steps[4])
    progress_bar.empty()

    return result


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------

def main():
    st.markdown(get_css(), unsafe_allow_html=True)
    _init_session_state()

    # Load production artifacts
    model, model_type = load_production_model()
    metadata = load_metadata()

    # Adapt sidebar to production model type
    view_mode, show_melspec, show_mfcc, show_temporal = _render_sidebar(model_type or "boosting")

    # Center main workspace layout
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────────────────────────────
    hcol1, hcol2 = st.columns([3.2, 0.8])
    with hcol1:
        render_header()
    with hcol2:
        st.markdown(
            "<div style='text-align:right; padding-top:14px;'>"
            "<span style='font-size:11px; color:#737E8C; text-transform:uppercase;"
            " letter-spacing:0.08em; font-weight:600;'>RESEARCH PROTOTYPE</span></div>",
            unsafe_allow_html=True,
        )
        if metadata:
            st.markdown(
                f"<div style='text-align:right; font-size:12px; color:#A7B0BC; font-weight:500;'>"
                f"v1.0</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:#2A3441; margin: 4px 0 20px;'>", unsafe_allow_html=True)

    # ── Workspace Columns ──────────────────────────────────────────────────
    col_left, col_right = st.columns([9, 11], gap="medium")

    with col_left:
        render_instructions()

    with col_right:
        st.markdown(f"### {copy.UPLOAD_TITLE}")
        
        # Streamlit standard drag-and-drop uploader
        uploaded_file = st.file_uploader(
            copy.UPLOAD_HELP, type=["wav", "WAV"],
            label_visibility="collapsed",
        )

        audio_bytes = None
        file_valid = False
        blocking_msg = None

        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            
            # Strict file size check
            if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
                blocking_msg = "File exceeds maximum size of 25 MB."
                file_valid = False
            else:
                file_valid = True

            if file_valid:
                current_hash = _audio_hash(audio_bytes)
                if current_hash != st.session_state["audio_hash"]:
                    st.session_state["audio_hash"] = current_hash
                    st.session_state["audio_bytes"] = audio_bytes
                    _clear_analysis_state()

                # Process basic metadata immediately
                try:
                    import soundfile as sf
                    with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
                        orig_sr = f.samplerate
                        orig_ch = f.channels
                        duration = len(f) / orig_sr
                except Exception:
                    duration = 0.0
                    orig_sr = 16000
                    orig_ch = 1

                # Quality precheck on original file
                from src.audio.quality import assess_audio_quality
                from src.audio.loader import load_audio
                from src.audio.preprocessing import preprocess_audio
                
                try:
                    raw_wav, l_sr, _, _, _ = load_audio(io.BytesIO(audio_bytes))
                    proc_wav, _ = preprocess_audio(raw_wav, l_sr)
                    quality_res = assess_audio_quality(proc_wav, l_sr)
                    errors = quality_res.get("blocking_errors", [])
                    quality_status = quality_res.get("quality_status", "Poor")
                except Exception as e:
                    errors = ["Could not parse audio header"]
                    quality_status = "Poor"

                # Check blockings
                if len(errors) > 0:
                    file_valid = False
                    blocking_msg = f"Quality gate blocked: {', '.join(errors)}"

                # Display summary card
                st.markdown("<div class='file-summary-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='file-title'>{uploaded_file.name}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='file-desc'>{duration:.2f} s · {orig_sr/1000:.1f} kHz · WAV</div>", unsafe_allow_html=True)
                
                q_color = "#4FAF83" if quality_status == "Good" else ("#C89B3C" if quality_status == "Acceptable" else "#D86464")
                st.markdown(
                    f"<div style='font-size:12px; color:#A7B0BC; margin-bottom:12px;'>"
                    f"Recording quality: <span style='color:{q_color}; font-weight:600;'>{quality_status}</span>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
                st.audio(audio_bytes, format="audio/wav")
                st.markdown("</div>", unsafe_allow_html=True)

            if not file_valid and blocking_msg:
                st.error(blocking_msg)

            # Analyze action button (Disabled if file invalid)
            analyze_btn = st.button(
                copy.BUTTON_ANALYZE,
                width="stretch",
                type="primary",
                disabled=not file_valid,
            )

            if analyze_btn and file_valid:
                result = _run_analysis(audio_bytes)
                st.session_state["prediction_result"] = result
                st.session_state["analysis_done"] = True
                st.rerun()

    st.markdown("<hr style='border-color:#2A3441; margin: 25px 0;'>", unsafe_allow_html=True)

    # ── Protocol Metadata Strip (Initial state only) ──────────────────────
    if not st.session_state["analysis_done"]:
        st.markdown(
            f"<div class='meta-strip'>"
            f"<div class='meta-item'>"
            f"  <div class='meta-label'>Expected Input</div>"
            f"  <div class='meta-value'>Sustained vowel /a/</div>"
            f"</div>"
            f"<div class='meta-item'>"
            f"  <div class='meta-label'>Recommended Duration</div>"
            f"  <div class='meta-value'>5–10 seconds</div>"
            f"</div>"
            f"<div class='meta-item'>"
            f"  <div class='meta-label'>Analysis Type</div>"
            f"  <div class='meta-value'>Acoustic ML</div>"
            f"</div>"
            f"<div class='meta-item'>"
            f"  <div class='meta-label'>Workstation Model</div>"
            f"  <div class='meta-value'>{metadata.get('production_model', '—') if metadata else '—'}</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Results & Tabs ─────────────────────────────────────────────────────
    if st.session_state["analysis_done"] and st.session_state["prediction_result"]:
        result = st.session_state["prediction_result"]
        proc_wav = st.session_state["processed_waveform"]
        proc_sr  = st.session_state["processed_sr"] or AUDIO_SAMPLE_RATE
        feats    = st.session_state.get("acoustic_feats") or {}

        render_result_summary(result)

        if "error" not in result:
            tabs = st.tabs(["Overview", "Signal Analysis", "Model Insights", "Research"])

            # ── TAB 1: Overview ─────────────────────────────────────────────
            with tabs[0]:
                st.plotly_chart(
                    create_score_scale(result["score"], result["threshold"]),
                    width="stretch",
                    config={"displayModeBar": False}
                )

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                dur  = result.get("audio_metrics", {}).get("processed_duration", 0)
                vfr  = result.get("quality", {}).get("metrics", {}).get("voiced_frame_ratio", 0)
                rms  = result.get("audio_metrics", {}).get("rms_after", 0)
                f0m  = feats.get("f0_mean", 0)
                thr  = result["threshold"]
                tms  = result["inference_time_ms"]

                m1.metric("Duration", f"{dur:.2f} s")
                m2.metric("Voiced frames", f"{vfr:.0%}")
                m3.metric("Mean F0", f"{f0m:.1f} Hz")
                m4.metric("RMS energy", f"{rms:.4f}")
                m5.metric("Threshold", f"{thr:.2f}")
                m6.metric("Inference", f"{tms} ms")

                if view_mode == "Standard":
                    st.markdown("---")
                    if proc_wav is not None:
                        st.plotly_chart(
                            create_waveform_chart(proc_wav, proc_sr),
                            width="stretch",
                            config={"displayModeBar": False}
                        )
                        st.plotly_chart(
                            create_f0_chart(proc_wav, proc_sr),
                            width="stretch",
                            config={"displayModeBar": False}
                        )


            # ── TAB 2: Signal Analysis ──────────────────────────────────────
            with tabs[1]:
                if view_mode == "Standard":
                    st.info("Detailed Analysis view is required to view signal graphs. Switch to Detailed or Research view in the sidebar to unlock.")
                else:
                    if proc_wav is not None:
                        st.plotly_chart(
                            create_waveform_chart(proc_wav, proc_sr),
                            width="stretch",
                            config={"displayModeBar": False}
                        )
                        st.plotly_chart(
                            create_f0_chart(proc_wav, proc_sr),
                            width="stretch",
                            config={"displayModeBar": False}
                        )

                        if show_melspec:
                            with st.spinner("Rendering Mel Spectrogram..."):
                                st.plotly_chart(
                                    create_mel_spectrogram(proc_wav, proc_sr),
                                    width="stretch",
                                    config={"displayModeBar": False}
                                )

                        if show_mfcc:
                            with st.spinner("Rendering MFCC Heatmap..."):
                                st.plotly_chart(
                                    create_mfcc_heatmap(proc_wav, proc_sr),
                                    width="stretch",
                                    config={"displayModeBar": False}
                                )

            # ── TAB 3: Model Insights ───────────────────────────────────────
            with tabs[2]:
                if view_mode == "Standard":
                    st.info("Detailed Analysis view is required to view model insights. Switch to Detailed or Research view in the sidebar to unlock.")
                else:
                    model_name = result.get("model_name", "")

                    if model_name == "CNN-BiLSTM" and model_type == "cnn_bilstm":
                        st.markdown("#### Temporal Model Attention")
                        st.caption(copy.TEMPORAL_EXPLANATION)

                        if show_temporal and proc_wav is not None:
                            try:
                                from src.explainability.temporal_saliency import get_temporal_importance
                                from src.audio.temporal_features import extract_temporal_features
                                cnn_model, _ = load_production_model()
                                norm_stats = load_norm_stats()

                                if cnn_model is not None and norm_stats is not None:
                                    with st.spinner("Extracting temporal attention..."):
                                        mfcc = extract_temporal_features(proc_wav, proc_sr)
                                        mfcc_norm = (mfcc - norm_stats["mean"]) / norm_stats["std"]
                                        mfcc_batch = np.expand_dims(mfcc_norm, 0)

                                        saliency = get_temporal_importance(
                                            cnn_model, mfcc_batch,
                                            hop_length=512, sample_rate=proc_sr,
                                        )

                                    if saliency:
                                        st.plotly_chart(
                                            create_attention_overlay(
                                                proc_wav,
                                                saliency["attention_weights"],
                                                saliency["frame_times"],
                                                saliency["top_intervals"],
                                                sr=proc_sr,
                                            ),
                                            width="stretch",
                                            config={"displayModeBar": False}
                                        )
                            except Exception:
                                st.info("Attention mapping not available.")
                    else:
                        # Classical model (SVM / Gradient Boosting)
                        st.markdown("#### Feature Model Influence (SHAP)")
                        st.caption(copy.SHAP_EXPLANATION)

                        try:
                            from src.explainability.feature_importance import (
                                compute_shap_values, build_top_influences, SHAP_AVAILABLE,
                            )

                            if not SHAP_AVAILABLE:
                                st.info("SHAP explainers are not active in this workspace.")
                            elif feats:
                                with st.spinner("Calculating SHAP influence..."):
                                    c_model, mtype = load_production_model()
                                    schema = load_feature_schema()
                                    bg_data = load_training_features()

                                    if c_model is not None and schema:
                                        feat_vec = np.array([[feats.get(k, 0.0) for k in schema]])
                                        shap_result = compute_shap_values(
                                            c_model, feat_vec, background_data=bg_data,
                                            model_type=mtype,
                                        )

                                        if shap_result and "shap_values" in shap_result and "error" not in shap_result:
                                            top_df = build_top_influences(
                                                shap_result["shap_values"], schema, top_n=10,
                                            )
                                            st.plotly_chart(
                                                create_shap_chart(top_df),
                                                width="stretch",
                                                config={"displayModeBar": False}
                                            )
                                            
                                            with st.expander("Acoustic Contribution Details", expanded=False):
                                                cols = ["label", "feature", "shap_value", "direction"] if view_mode == "Research" else ["label", "shap_value", "direction"]
                                                col_renames = {"label": "Feature", "feature": "Internal Identifier", "shap_value": "Model Influence Score", "direction": "Direction"}
                                                st.dataframe(
                                                    top_df[cols].rename(columns=col_renames),
                                                    width="stretch",
                                                )
                        except Exception:
                            st.info("Feature influence view not available.")

            # ── TAB 4: Research ─────────────────────────────────────────────
            with tabs[3]:
                if view_mode != "Research":
                    st.info("Research view is required to view raw tables and validation metrics. Switch to Research view in the sidebar to unlock.")
                else:
                    if metadata:
                        r1, r2 = st.columns(2)
                        with r1:
                            st.markdown("#### Dataset")
                            st.markdown(f"**Source:** {metadata.get('dataset_source', '—')}")
                            st.markdown(f"**Total recordings:** {metadata.get('dataset_file_count', '—')}")
                            st.markdown(f"**Healthy Control:** {metadata.get('healthy_sample_count', '—')}")
                            st.markdown(f"**Parkinson's:** {metadata.get('parkinsons_sample_count', '—')}")
                            st.markdown(f"**Unique speakers:** {metadata.get('unique_speaker_count', '—')}")
                            st.markdown(f"**Voice task:** {metadata.get('expected_voice_task', '—')}")

                        with r2:
                            st.markdown("#### Model")
                            st.markdown(f"**Production model:** {metadata.get('production_model', '—')}")
                            st.markdown(f"**Input representation:** {metadata.get('input_representation', '—')}")
                            st.markdown(f"**Sample rate:** {metadata.get('sample_rate', '—')} Hz")
                            st.markdown(f"**Version:** {metadata.get('project_version', '—')}")
                            st.markdown(f"**Training date:** {metadata.get('training_date', '—')[:10]}")

                        st.markdown("---")
                        st.markdown("#### Evaluation")
                        vm = metadata.get("validation_metrics", {})
                        e1, e2, e3, e4, e5, e6 = st.columns(6)
                        e1.metric("CV strategy", metadata.get("validation_strategy", "—"))
                        e2.metric("Folds", metadata.get("cross_validation_folds", "—"))
                        e3.metric("Threshold", f"{metadata.get('selected_threshold', 0):.2f}")
                        e4.metric("Mean Acc.", f"{vm.get('mean_accuracy', 0):.3f}")
                        e5.metric("Mean AUC", f"{vm.get('mean_roc_auc', 0):.3f}")
                        e6.metric("Mean F1", f"{vm.get('mean_f1', 0):.3f}")

                        st.markdown("---")
                        st.markdown("#### Model Comparison")
                        comp_path = REPORTS_DIR / "model_comparison.csv"
                        if comp_path.exists():
                            comp_df = pd.read_csv(comp_path)
                            prod_model = metadata.get("production_model", "")

                            def _highlight_prod(row):
                                if row.get("model", "") == prod_model:
                                    return ["background-color: rgba(91,141,239,0.12)"] * len(row)
                                return [""] * len(row)

                            st.dataframe(
                                comp_df.style.apply(_highlight_prod, axis=1),
                                width="stretch",
                            )
                        st.json(metadata)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
