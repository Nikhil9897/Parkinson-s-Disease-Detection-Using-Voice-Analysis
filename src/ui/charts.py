"""
src/ui/charts.py

All Plotly chart builders for the NeuroVoice AI interface.

Design principles:
  - Transparent / dark-compatible backgrounds
  - Subtle grid lines (rgba(167,176,188,0.10))
  - Minimal toolbar / no unnecessary decorations
  - Responsive via use_container_width=True in the caller
  - Downsampled waveforms (max 3,000–5,000 points) to keep rendering fast
"""

import plotly.graph_objects as go
import numpy as np
import librosa
import pandas as pd
from typing import Optional, List, Tuple

from config import AUDIO_SAMPLE_RATE, HOP_LENGTH, N_FFT, N_MFCC

# Shared layout defaults
_GRID   = "rgba(167, 176, 188, 0.10)"
_PAPER  = "rgba(0,0,0,0)"
_PLOT   = "rgba(0,0,0,0)"
_TEXT   = "#A7B0BC"
_MARGIN = dict(l=50, r=20, t=45, b=45)


# ---------------------------------------------------------------------------
# Score scale
# ---------------------------------------------------------------------------

def create_score_scale(score: float, threshold: float) -> go.Figure:
    """
    Horizontal probability scale with threshold marker and score dot.

    Lower                 Intermediate               Elevated
    0% ─────────────────────────────────────────── 100%
                                ● 0.72
    """
    fig = go.Figure()

    # Background track
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0],
        mode="lines",
        line=dict(color="rgba(167,176,188,0.20)", width=6),
        hoverinfo="skip", showlegend=False,
    ))

    # Threshold dashed marker
    fig.add_shape(type="line",
        x0=threshold, x1=threshold, y0=-0.6, y1=0.6,
        line=dict(color="#A7B0BC", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=threshold, y=0.75, text=f"Threshold {threshold:.2f}",
        font=dict(color="#A7B0BC", size=11), showarrow=False,
    )

    # Region labels
    for x, label in [(0.07, "Lower"), (0.50, "Intermediate"), (0.88, "Elevated")]:
        fig.add_annotation(
            x=x, y=-0.85, text=label,
            font=dict(color=_TEXT, size=11), showarrow=False,
        )

    # Score dot
    if score < threshold - 0.1:
        color = "#4FAF83"
    elif score > threshold + 0.1:
        color = "#D86464"
    else:
        color = "#C89B3C"

    fig.add_trace(go.Scatter(
        x=[score], y=[0],
        mode="markers+text",
        marker=dict(color=color, size=18, line=dict(color="#0D1117", width=2)),
        text=[f"{score:.2f}"],
        textposition="top center",
        textfont=dict(color=color, size=14, family="Inter, sans-serif"),
        name="Score", hoverinfo="x",
    ))

    fig.update_layout(
        xaxis=dict(
            range=[-0.05, 1.05], showgrid=False, zeroline=False,
            tickvals=[0, 0.25, 0.50, 0.75, 1.0],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            color=_TEXT,
        ),
        yaxis=dict(range=[-1.1, 1.1], showgrid=False, zeroline=False, showticklabels=False),
        height=130,
        margin=dict(l=20, r=20, t=30, b=40),
        plot_bgcolor=_PLOT, paper_bgcolor=_PAPER,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------

def create_waveform_chart(waveform: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> go.Figure:
    """Downsampled waveform chart. Target ≤ 4,000 display points."""
    target = 4000
    if len(waveform) > target:
        step = len(waveform) // target
        disp = waveform[::step]
        time = np.arange(len(disp)) * step / sr
    else:
        disp = waveform
        time = np.arange(len(waveform)) / sr

    fig = go.Figure(go.Scatter(
        x=time, y=disp,
        line=dict(color="#5B8DEF", width=0.8),
        hovertemplate="t=%{x:.3f}s<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Waveform", font=dict(color=_TEXT, size=13)),
        height=230,
        margin=_MARGIN,
        plot_bgcolor=_PLOT,
        paper_bgcolor=_PAPER,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="Time (s)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        yaxis=dict(
            title="Amplitude",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Fundamental Frequency
# ---------------------------------------------------------------------------

def create_f0_chart(waveform: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> go.Figure:
    """F0 scatter plot — only voiced frames are plotted (no misleading connections)."""
    f0, voiced_flag, _ = librosa.pyin(
        waveform,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=2048,
        hop_length=HOP_LENGTH,
        fill_na=None,
    )
    time = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    valid = ~np.isnan(f0)

    voiced_f0 = f0[valid]
    mean_f0 = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time[valid], y=f0[valid],
        mode="markers",
        marker=dict(color="#4FAF83", size=3.5, opacity=0.85),
        name="F0 (voiced)",
        hovertemplate="t=%{x:.3f}s  F0=%{y:.1f}Hz<extra></extra>",
    ))

    if mean_f0 > 0:
        fig.add_hline(y=mean_f0, line=dict(color="#C89B3C", width=1.2, dash="dot"),
                      annotation_text=f"Mean {mean_f0:.1f} Hz",
                      annotation_font=dict(color="#C89B3C", size=11))

    fig.update_layout(
        title=dict(text="Fundamental Frequency (F0) — voiced frames only", font=dict(color=_TEXT, size=13)),
        height=230,
        margin=_MARGIN,
        plot_bgcolor=_PLOT,
        paper_bgcolor=_PAPER,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="Time (s)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        yaxis=dict(
            title="Frequency (Hz)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Mel Spectrogram
# ---------------------------------------------------------------------------

def create_mel_spectrogram(waveform: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> go.Figure:
    """
    Mel-scale spectrogram using a perceptually appropriate colorscale.
    Uses 'Blues' reversed — avoids rainbow colormaps.
    Computed once and rendered as a Heatmap for performance.
    """
    S = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=80, fmax=8000,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    n_frames = S_db.shape[1]
    duration = len(waveform) / sr
    time_axis = np.linspace(0, duration, n_frames)
    mel_freqs = librosa.mel_frequencies(n_mels=80, fmax=8000)

    fig = go.Figure(go.Heatmap(
        z=S_db,
        x=time_axis,
        y=mel_freqs.astype(int),
        colorscale="Blues",
        reversescale=True,
        colorbar=dict(
            title=dict(text="dB", font=dict(color=_TEXT, size=11)),
            tickfont=dict(color=_TEXT, size=10),
        ),
        hovertemplate="t=%{x:.3f}s  %{y}Hz  %{z:.1f}dB<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Mel Spectrogram", font=dict(color=_TEXT, size=13)),
        height=280,
        margin=_MARGIN,
        plot_bgcolor=_PLOT,
        paper_bgcolor=_PAPER,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="Time (s)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        yaxis=dict(
            title="Frequency (Hz)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# MFCC Heatmap
# ---------------------------------------------------------------------------

def create_mfcc_heatmap(waveform: np.ndarray, sr: int = AUDIO_SAMPLE_RATE) -> go.Figure:
    """
    MFCC coefficient heatmap over time.
    40 coefficients × time frames. Colorscale: RdBu (diverging from zero).
    """
    mfccs = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    n_frames = mfccs.shape[1]
    duration = len(waveform) / sr
    time_axis = np.linspace(0, duration, n_frames)
    coeff_labels = [f"MFCC {i+1}" for i in range(N_MFCC)]

    # Downsample time axis for display if very long
    if n_frames > 300:
        step = n_frames // 300
        mfccs_disp = mfccs[:, ::step]
        time_disp = time_axis[::step]
    else:
        mfccs_disp = mfccs
        time_disp = time_axis

    fig = go.Figure(go.Heatmap(
        z=mfccs_disp,
        x=time_disp,
        y=coeff_labels,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(
            title=dict(text="Value", font=dict(color=_TEXT, size=11)),
            tickfont=dict(color=_TEXT, size=10),
        ),
        hovertemplate="t=%{x:.3f}s  %{y}  %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="MFCC Coefficients (40)", font=dict(color=_TEXT, size=13)),
        height=340,
        margin=_MARGIN,
        plot_bgcolor=_PLOT,
        paper_bgcolor=_PAPER,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="Time (s)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, color=_TEXT,
            tickfont=dict(size=9),
            autorange="reversed",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# SHAP Feature Importance
# ---------------------------------------------------------------------------

def create_shap_chart(influences_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of top SHAP model influences.

    Parameters
    ----------
    influences_df : DataFrame
        From feature_importance.build_top_influences().
        Must have columns: label, shap_value, direction.
    """
    df = influences_df.copy().sort_values("shap_value", ascending=True)

    colors = ["#4FAF83" if v < 0 else "#D86464" for v in df["shap_value"]]

    fig = go.Figure(go.Bar(
        x=df["shap_value"],
        y=df["label"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=df["direction"],
        textposition="outside",
        textfont=dict(color=_TEXT, size=10),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color="#A7B0BC", width=1))

    fig.update_layout(
        title=dict(text="Model Influence (SHAP)", font=dict(color=_TEXT, size=13)),
        height=max(280, len(df) * 32 + 60),
        margin=_MARGIN,
        plot_bgcolor=_PLOT,
        paper_bgcolor=_PAPER,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="SHAP value (model influence direction and magnitude)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        yaxis=dict(showgrid=False, zeroline=False, color=_TEXT, autorange=True),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Temporal Attention Overlay
# ---------------------------------------------------------------------------

def create_attention_overlay(
    waveform: np.ndarray,
    attention_weights: np.ndarray,
    frame_times: np.ndarray,
    top_intervals: List[Tuple[float, float, float]],
    sr: int = AUDIO_SAMPLE_RATE,
) -> go.Figure:
    """
    Waveform with temporal attention weight overlay.

    Displays the downsampled waveform and shades regions
    with high model attention weights.
    """
    # Downsample waveform for display
    target = 3000
    if len(waveform) > target:
        step = len(waveform) // target
        disp_wav = waveform[::step]
        time_wav = np.arange(len(disp_wav)) * step / sr
    else:
        disp_wav = waveform
        time_wav = np.arange(len(waveform)) / sr

    fig = go.Figure()

    # Shade high-attention intervals
    max_w = float(np.max(attention_weights)) if len(attention_weights) > 0 else 1.0
    for (start, end, w) in top_intervals:
        opacity = max(0.1, min(0.4, w / max_w * 0.5))
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="#5B8DEF", opacity=opacity,
            layer="below", line_width=0,
        )

    # Waveform
    fig.add_trace(go.Scatter(
        x=time_wav, y=disp_wav,
        line=dict(color="#5B8DEF", width=0.8),
        name="Waveform",
        hovertemplate="t=%{x:.3f}s<extra></extra>",
    ))

    # Attention weight curve (normalized, secondary y-axis)
    if len(frame_times) > 0:
        norm_attn = attention_weights / (max_w + 1e-8)
        fig.add_trace(go.Scatter(
            x=frame_times, y=norm_attn,
            line=dict(color="#C89B3C", width=1.2, dash="dot"),
            name="Attention weight",
            yaxis="y2",
            hovertemplate="t=%{x:.3f}s  attn=%{y:.3f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text="Waveform + Temporal Model Attention", font=dict(color=_TEXT, size=13)),
        height=270,
        margin=_MARGIN,
        plot_bgcolor=_PLOT,
        paper_bgcolor=_PAPER,
        font=dict(color=_TEXT, size=12),
        xaxis=dict(
            title="Time (s)",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        yaxis=dict(
            title="Amplitude",
            showgrid=True, gridcolor=_GRID, zeroline=False, color=_TEXT,
        ),
        yaxis2=dict(
            title="Attention weight (normalized)",
            overlaying="y", side="right",
            showgrid=False, color=_TEXT,
            range=[0, 1.2],
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=_TEXT, size=11),
        ),
    )
    return fig
