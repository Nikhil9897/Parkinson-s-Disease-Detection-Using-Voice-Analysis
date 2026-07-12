def get_css() -> str:
    return """
<style>
    /* Base typography and background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #F2F5F8;
    }
    
    .stApp {
        background-color: #0D1117;
    }
    
    /* Streamlit Main View Block Constraint to ~1400px max width */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 1400px !important;
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Style Streamlit Native File Uploader to match spec */
    [data-testid="stFileUploader"] section {
        background-color: #151B23 !important;
        border: 1px solid #2A3441 !important;
        border-radius: 8px !important;
        padding: 18px !important;
        transition: border-color 0.2s, background-color 0.2s;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #5B8DEF !important;
        background-color: #18202B !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: #F2F5F8;
    }
    
    /* Main Content Constraints */
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 10px;
    }
    
    /* Header layout */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding-bottom: 5px;
    }
    
    .title-text {
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 0px;
        padding-bottom: 0px;
        color: #F2F5F8;
    }
    
    .subtitle-text {
        font-size: 15px;
        color: #A7B0BC;
        margin-top: 4px;
        margin-bottom: 2px;
    }
    
    .disclaimer-text {
        font-size: 13px;
        color: #737E8C;
        margin-top: 2px;
        margin-bottom: 10px;
    }
    
    /* Static Waveform Motif SVG styling */
    .waveform-motif {
        stroke: #2A3441;
        fill: none;
        stroke-width: 1.5;
        vertical-align: middle;
        margin-right: 12px;
        display: inline-block;
    }
    
    /* System & Metadata Rows */
    .meta-strip {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #151B23;
        border: 1px solid #2A3441;
        border-radius: 8px;
        padding: 12px 24px;
        margin: 15px 0;
    }
    
    .meta-item {
        flex: 1;
        text-align: center;
        border-right: 1px solid #2A3441;
    }
    
    .meta-item:last-child {
        border-right: none;
    }
    
    .meta-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #737E8C;
        margin-bottom: 3px;
    }
    
    .meta-value {
        font-size: 14px;
        font-weight: 500;
        color: #F2F5F8;
    }
    
    /* Voice Test Instruction Row */
    .instruction-row {
        display: flex;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    
    .instruction-num {
        font-size: 12px;
        font-weight: 600;
        color: #737E8C;
        margin-right: 12px;
        margin-top: 2px;
        letter-spacing: 0.05em;
    }
    
    .instruction-content {
        font-size: 14px;
        color: #A7B0BC;
    }
    
    /* Custom Upload Box wrapper */
    .custom-upload-box {
        background-color: #151B23;
        border: 1px dashed #2A3441;
        border-radius: 8px;
        padding: 24px;
        text-align: center;
        transition: border-color 0.2s, background-color 0.2s;
    }
    
    .custom-upload-box:hover {
        border-color: #5B8DEF;
        background-color: #18202B;
    }
    
    .upload-icon {
        color: #737E8C;
        margin-bottom: 8px;
    }
    
    .upload-main-text {
        font-size: 15px;
        font-weight: 500;
        color: #F2F5F8;
        margin-bottom: 4px;
    }
    
    .upload-sub-text {
        font-size: 13px;
        color: #A7B0BC;
        margin-bottom: 12px;
    }
    
    .upload-footer-text {
        font-size: 11px;
        color: #737E8C;
    }

    /* File summary */
    .file-summary-card {
        background-color: #151B23;
        border: 1px solid #2A3441;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }

    .file-title {
        font-size: 14px;
        font-weight: 600;
        color: #F2F5F8;
        margin-bottom: 2px;
    }

    .file-desc {
        font-size: 12px;
        color: #A7B0BC;
        margin-bottom: 8px;
    }
    
    /* Result Summary Area */
    .result-summary {
        background-color: #151B23;
        border: 1px solid #2A3441;
        border-radius: 8px;
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Cohesive wrapper for results columns block */
    div:has(> div[data-testid="stMarkdownContainer"] > #result-summary-marker) + div[data-testid="stHorizontalBlock"] {
        background-color: #151B23 !important;
        border: 1px solid #2A3441 !important;
        border-radius: 8px !important;
        padding: 24px !important;
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }
    
    .score-value {
        font-size: 48px;
        font-weight: 600;
        line-height: 1;
        color: #F2F5F8;
    }
    
    .score-label {
        font-size: 14px;
        color: #A7B0BC;
        margin-top: 5px;
    }
    
    .score-caption {
        font-size: 12px;
        color: #737E8C;
    }
    
    .band-indicator-lower {
        color: #4FAF83;
        font-weight: 600;
        font-size: 20px;
    }
    .band-indicator-intermediate {
        color: #C89B3C;
        font-weight: 600;
        font-size: 20px;
    }
    .band-indicator-elevated {
        color: #D86464;
        font-weight: 600;
        font-size: 20px;
    }
    
    /* Buttons */
    .stButton > button[type="secondary"] {
        background-color: transparent !important;
        color: #737E8C !important;
        border: 1px solid #2A3441 !important;
    }
    
    .stButton > button {
        background-color: #5B8DEF !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background-color: #6C9AF2 !important;
        color: #FFFFFF !important;
    }
    
    /* Accent overrides for Streamlit controls to use #5B8DEF instead of red/orange */
    /* Radio buttons selection color override */
    div[data-baseweb="radio"] div[role="presentation"] {
        border-color: #5B8DEF !important;
    }
    div[data-baseweb="radio"] input:checked + div {
        background-color: #5B8DEF !important;
        border-color: #5B8DEF !important;
    }
    
    /* Checkbox selection override */
    div[data-baseweb="checkbox"] div[role="presentation"] {
        border-color: #5B8DEF !important;
    }
    div[data-baseweb="checkbox"] input:checked + div {
        background-color: #5B8DEF !important;
        border-color: #5B8DEF !important;
    }
    
    /* Slider primary color override */
    div[data-testid="stSlider"] div[role="slider"] {
        background-color: #5B8DEF !important;
    }
    
    /* Tabs active accent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #2A3441;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #A7B0BC;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #5B8DEF !important;
        color: #5B8DEF !important;
    }
    
    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #2A3441;
        background-color: #0E131A;
    }
    
    section[data-testid="stSidebar"] h3 {
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #737E8C;
        margin-top: 24px;
        margin-bottom: 8px;
    }
    
</style>
"""
