import streamlit as st
from src.ui import copy

def render_header():
    # Subtle static waveform SVG motif
    svg_waveform = (
        '<svg class="waveform-motif" width="36" height="22" viewBox="0 0 36 22">'
        '<path d="M2,11 L6,11 L10,3 L14,19 L18,6 L22,16 L26,11 L30,11 L34,11" />'
        '</svg>'
    )
    
    st.markdown(
        f"<div class='title-text'>{svg_waveform}{copy.APP_TITLE}</div>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='subtitle-text'>{copy.APP_SUBTITLE}</div>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='disclaimer-text'>{copy.DISCLAIMER_RESEARCH}</div>", 
        unsafe_allow_html=True
    )

def render_instructions():
    st.markdown(
        "<div style='font-size: 11px; font-weight: 600; color: #737E8C; "
        "letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px;'>"
        "Sustained /a/ Protocol</div>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<h3 style='margin-top: 0px; margin-bottom: 8px;'>{copy.INSTRUCTIONS_TITLE}</h3>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size: 14px; color: #A7B0BC; margin-bottom: 20px;'>"
        f"Sustain the vowel <span style='color: #5B8DEF; font-weight: 500;'>\"A\"</span> "
        f"continuously for <span style='color: #5B8DEF; font-weight: 500;'>5–10 seconds</span>.</p>", 
        unsafe_allow_html=True
    )
    
    # Custom numbered rows
    instructions = [
        "Quiet environment",
        "Natural speaking volume",
        "Microphone 15–20 cm away",
        "Continuous \"AAAA\" sound"
    ]
    
    for idx, text in enumerate(instructions, 1):
        num_str = f"{idx:02d}"
        st.markdown(
            f"<div class='instruction-row'>"
            f"<div class='instruction-num'>{num_str}</div>"
            f"<div class='instruction-content'>{text}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

def render_quality_badge(quality: str):
    if quality == copy.QUALITY_GOOD:
        color = "#4FAF83"
    elif quality == copy.QUALITY_ACCEPT:
        color = "#C89B3C"
    else:
        color = "#D86464"
    st.markdown(f"<span style='color:{color}; font-weight:600;'>{quality}</span>", unsafe_allow_html=True)
    
def render_result_summary(prediction_result: dict):
    if "error" in prediction_result:
        st.error(prediction_result["error"])
        if "quality" in prediction_result:
            st.error(f"Quality Errors: {prediction_result['quality'].get('blocking_errors', [])}")
        return
        
    score = prediction_result["score"]
    band = prediction_result["pattern_band"]
    quality = prediction_result["quality"]["quality_status"]
    time_ms = prediction_result["inference_time_ms"]
    
    # CSS :has target marker for cohesive container border/bg styling
    st.markdown("<div id='result-summary-marker'></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.1, 1.8, 1.1])
    
    with col1:
        st.markdown(f"<div class='score-value'>{score:.2f}</div>", unsafe_allow_html=True)
        st.markdown("<div class='score-label'>Parkinsonian voice pattern score</div>", unsafe_allow_html=True)
        st.markdown("<div class='score-caption'>Model probability · Not diagnostic confidence</div>", unsafe_allow_html=True)
        
    with col2:
        if band == copy.BAND_LOWER:
            band_class = "band-indicator-lower"
            marker = "<span style='color: #4FAF83; margin-right: 8px;'>●</span>"
            interp = copy.INTERP_LOWER
        elif band == copy.BAND_INTERMEDIATE:
            band_class = "band-indicator-intermediate"
            marker = "<span style='color: #C89B3C; margin-right: 8px;'>●</span>"
            interp = copy.INTERP_INTERMEDIATE
        else:
            band_class = "band-indicator-elevated"
            marker = "<span style='color: #D86464; margin-right: 8px;'>●</span>"
            interp = copy.INTERP_ELEVATED
            
        st.markdown(f"<div class='{band_class}'>{marker}{band}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 13.5px; color: #A7B0BC; line-height: 1.5; margin-top: 10px;'>{interp}</p>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='score-label' style='margin-top: 0px;'>Recording quality</div>", unsafe_allow_html=True)
        render_quality_badge(quality)
        st.markdown(f"<div class='score-caption' style='margin-top: 10px;'>Inference time: {time_ms} ms</div>", unsafe_allow_html=True)
