# streamlit_app.py
#old_version
import io
import os
import sys
import base64
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import backend as K
import torch
import gdown
import segmentation_models_pytorch as smp

st.set_page_config(
    page_title="Advanced Wound Analysis",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# ──── Config ───────────────────────────────────────────────────
# Original Sugar Heal Model (Wound Hole Segmentation)
SUGAR_MODEL_PATH = Path("unet_wound_segmentation_best.h5")
SUGAR_MODEL_URL  = "https://drive.google.com/uc?id=1_PToBgQjEKAQAZ9ZX10sRpdgxQ18C-18"

# Advanced Tissue Analysis Model
TISSUE_MODEL_PATH = Path("best_model_streamlit.pth")
TISSUE_MODEL_ID = "1q0xk9wll0eyF3-CKEc5s6MfG0gE_jde1"

LOGO_PATH  = Path("GREEN.png")
IMG_SIZE   = 256
THRESHOLD  = 0.5
ALPHA      = 0.4

# Tissue Analysis Config - Keep 9 classes to match the model
N_CLASSES = 9
ENCODER = "mit_b3"
CLASS_NAMES = [
    "background", "fibrin", "granulation", "callus", "necrotic", "eschar", "neodermis", "tendon", "dressing"
]

# Classes we'll actually display and use
DISPLAY_CLASSES = ["background", "fibrin", "granulation", "callus"]

# ──── CENTRALIZED COLOR CONTROL ────────────────────────────────────────────────
# Define tissue colors (RGB format for display)
TISSUE_COLORS_RGB = {
    "background": (0, 0, 0),         # Black
    "fibrin": (255, 255, 0),         # Yellow  
    "granulation": (255, 0, 0),      # Red
    "callus": (0, 0, 255),           # Blue
    "necrotic": (255, 165, 0),       # Orange - Not displayed
    "eschar": (128, 0, 128),         # Purple - Not displayed
    "neodermis": (0, 255, 255),      # Cyan - Not displayed
    "tendon": (255, 192, 203),       # Pink - Not displayed
    "dressing": (0, 255, 0),         # Green - Not displayed
}

# Convert to BGR for OpenCV processing
TISSUE_COLORS_BGR = {name: (color[2], color[1], color[0]) for name, color in TISSUE_COLORS_RGB.items()}

# Create palette array for model processing (BGR order)
PALETTE = [TISSUE_COLORS_BGR[name] for name in CLASS_NAMES]

# Color hex codes for HTML display
TISSUE_COLORS_HEX = {name: f"rgb({color[0]}, {color[1]}, {color[2]})" for name, color in TISSUE_COLORS_RGB.items()}

# Tissue health scoring weights
TISSUE_HEALTH_WEIGHTS = {
    "granulation": 0.8,    # Good healing tissue
    "fibrin": 0.6,         # Moderate - part of healing
    "callus": 0.4,         # Poor - hard tissue
    "background": 0.0      # Neutral
}

# Download models from Google Drive if not present
def download_models():
    if not SUGAR_MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        st.info("Downloading High accuracy segmentation model...")
        gdown.download(SUGAR_MODEL_URL, str(SUGAR_MODEL_PATH), quiet=False)

    if not TISSUE_MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        st.info("Downloading tissue analysis model...")
        gdown.download(f"https://drive.google.com/uc?id={TISSUE_MODEL_ID}", str(TISSUE_MODEL_PATH), quiet=False)

# Ensure models are available
download_models()

# ──── Dynamic Color Palette Based on Theme ───────────────────────────────────────────────────────
def get_theme_colors():
    if st.session_state.dark_mode:
        return {
            "primary"    : "#074225",
            "secondary"  : "#41706F",
            "accent"     : "#3B6C53",
            "dark"       : "#335F4B",
            "light"      : "#81A295",
            "surface"    : "#1a1a1a",
            "surface_secondary": "#2a2a2a",
            "text_primary" : "#E0E0E0",
            "text_secondary" : "#FFFFFF",
            "highlight"  : "rgb(122,164,140)",
            "success"    : "#28a745",
            "warning"    : "#ffc107",
            "danger"     : "#dc3545",
            "gradient_start": "#074225",
            "gradient_end": "#3B6C53",
            "card_bg": "rgba(59, 108, 83, 0.1)",
            "border_color": "rgba(122,164,140,0.2)",
        }
    else:
        return {
            "primary"    : "#0a5d35",
            "secondary"  : "#5a8d8c",
            "accent"     : "#4a8a6a",
            "dark"       : "#3d7a5f",
            "light"      : "#6fa98f",
            "surface"    : "#f8f9fa",
            "surface_secondary": "#ffffff",
            "text_primary" : "#2c3e50",
            "text_secondary" : "#1a1a1a",
            "highlight"  : "#2e7d32",
            "success"    : "#4caf50",
            "warning"    : "#ff9800",
            "danger"     : "#f44336",
            "gradient_start": "#0a5d35",
            "gradient_end": "#4a8a6a",
            "card_bg": "rgba(74, 138, 106, 0.05)",
            "border_color": "rgba(46, 125, 50, 0.2)",
        }

COL = get_theme_colors()

# Theme toggle button
col1, col2, col3 = st.columns([1, 8, 1])
with col3:
    if st.button("🌓", help="Toggle theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Enhanced CSS with theme support
st.markdown(f"""
<style>
  /* Base Styles */
  .stApp {{ 
    background: {"linear-gradient(135deg, " + COL['surface'] + " 0%, " + COL['surface_secondary'] + " 100%)" if st.session_state.dark_mode else COL['surface']}; 
    color: {COL['text_primary']}; 
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
  }}
  
  /* Header Styles with enhanced gradient and animations */
  .header {{ 
    text-align: center; 
    padding: 40px 30px; 
    background: linear-gradient(135deg, {COL['gradient_start']} 0%, {COL['gradient_end']} 50%, {COL['dark']} 100%); 
    color: {COL['text_secondary'] if st.session_state.dark_mode else '#ffffff'}; 
    border-radius: 20px; 
    box-shadow: 0 10px 30px rgba(0,0,0,0.2), 0 0 50px {COL['border_color']}; 
    margin-bottom: 40px; 
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
  }}
  
  .header::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    animation: shimmer 3s infinite;
  }}
  
  @keyframes shimmer {{
    0% {{ left: -100%; }}
    100% {{ left: 100%; }}
  }}
  
  .header h1 {{ 
    margin: 0; 
    font-size: 3rem; 
    font-weight: 800; 
    letter-spacing: 2px; 
    text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #ffffff, {COL['highlight']});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  
  .header p {{ 
    font-size: 1.3rem; 
    margin-top: 15px; 
    opacity: 0.95; 
    font-weight: 400;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    color: #ffffff !important;
  }}
  
  /* Enhanced Instructions Box */
  .instructions {{ 
    background: {"linear-gradient(145deg, " + COL['dark'] + " 0%, #2a4a37 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #e8f5e9 0%, #f1f8e9 100%)"}; 
    padding: 35px; 
    border-left: 8px solid {COL['highlight']}; 
    border-radius: 15px; 
    margin-bottom: 40px; 
    color: {COL['text_primary']}; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid {COL['border_color']};
  }}
  
  .instructions strong {{ 
    color: {COL['highlight']}; 
    font-size: 1.4rem; 
    text-shadow: 0 2px 4px rgba(0,0,0,0.15);
  }}
  
  .instructions ol {{ 
    padding-left: 35px; 
    margin-top: 20px; 
    counter-reset: item;
  }}
  
  .instructions li {{ 
    margin-bottom: 12px; 
    font-size: 1.15rem; 
    line-height: 1.6;
    counter-increment: item;
    position: relative;
    color: {COL['text_primary']};
  }}
  
  .instructions li::marker {{
    color: {COL['highlight']};
    font-weight: bold;
  }}
  
  /* Enhanced Logo Container */
  .logo-container {{
    background: linear-gradient(145deg, {COL['highlight']} 0%, #4a7a5c 100%); 
    padding: 25px; 
    border-radius: 20px; 
    text-align: center; 
    margin-bottom: 35px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.2);
    border: 2px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
  }}
  
  .logo-container:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.2);
  }}
  
  img.logo {{ 
    display: block; 
    margin: 0 auto; 
    width: 100%; 
    max-width: 1000px;
    padding: 10px; 
    transition: all 0.3s ease;
    filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
  }}
  
  /* Enhanced Button Styling */
  .stButton>button {{ 
    background: linear-gradient(135deg, {COL['gradient_start']} 0%, {COL['gradient_end']} 50%, {COL['dark']} 100%); 
    color: white; 
    border: none; 
    border-radius: 15px; 
    padding: 20px 40px; 
    font-weight: 700; 
    transition: all .4s ease; 
    box-shadow: 0 6px 20px rgba(0,0,0,0.15); 
    width: 100%;
    font-size: 1.3rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
  }}
  
  .stButton>button::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: all 0.5s;
  }}
  
  .stButton>button:hover::before {{
    left: 100%;
  }}
  
  .stButton>button:hover {{ 
    background: linear-gradient(135deg, {COL['accent']} 0%, {COL['gradient_start']} 50%, {COL['primary']} 100%); 
    transform: translateY(-4px); 
    box-shadow: 0 10px 30px rgba(0,0,0,0.25), 0 0 20px {COL['border_color']}; 
  }}
  
  /* Enhanced File Uploader */
  .css-1cpxqw2, [data-testid="stFileUploader"] {{ 
    border: 3px dashed {COL['accent']}; 
    background: {COL['card_bg']}; 
    border-radius: 20px; 
    padding: 35px; 
    transition: all 0.4s ease;
    backdrop-filter: blur(10px);
  }}
  
  .css-1cpxqw2:hover, [data-testid="stFileUploader"]:hover {{ 
    border-color: {COL['highlight']}; 
    background: {"linear-gradient(145deg, rgba(59, 108, 83, 0.2), rgba(59, 108, 83, 0.3))" if st.session_state.dark_mode else "linear-gradient(145deg, rgba(74, 138, 106, 0.1), rgba(74, 138, 106, 0.15))"}; 
    box-shadow: 0 8px 25px {COL['border_color']};
    transform: translateY(-2px);
  }}
  
  /* Enhanced Image Container */
  .img-container {{ 
    background: {"linear-gradient(145deg, " + COL['dark'] + " 0%, #2a4a37 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #f5f5f5 0%, #ffffff 100%)"}; 
    padding: 30px; 
    border-radius: 20px; 
    box-shadow: 0 8px 25px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.1); 
    margin-bottom: 15px; 
    transition: all 0.4s ease;
    overflow: hidden;
    text-align: center;
    height: 100%;
    display: flex !important;
    flex-direction: column;
    justify-content: center;
    align-items: center !important;
    width: 100% !important;
    border: 1px solid {COL['border_color']};
  }}
  
  .img-container:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 35px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.1);
  }}
  
  /* Enhanced Image Styling */
  .img-container img,
  .stImage img {{ 
    max-height: 1400px !important;
    max-width: 100% !important;
    width: auto !important; 
    height: auto !important;
    margin: 0 auto !important; 
    display: block !important; 
    border-radius: 12px !important;
    transition: all 0.4s ease;
    object-fit: contain !important;
    filter: drop-shadow(0 8px 16px rgba(0,0,0,0.15));
  }}
  
  .img-container img:hover,
  .stImage img:hover {{
    transform: scale(1.02);
    filter: drop-shadow(0 12px 24px rgba(0,0,0,0.2));
  }}
  
  /* Center all Streamlit images */
  [data-testid="stImage"] {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
  }}
  
  /* Enhanced Image Captions */
  .img-container figcaption, .stImage figcaption, .css-1b0udgb, .css-83jbox {{
    font-size: 1.4rem !important;
    color: {COL['highlight']} !important;
    margin-top: 20px !important;
    font-weight: 700 !important;
    text-align: center !important;
    width: 100% !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    letter-spacing: 0.5px;
  }}
  
  figcaption p {{
    font-size: 1.4rem !important;
    margin: 15px 0 !important;
    color: {COL['highlight']} !important;
    text-align: center !important;
    font-weight: 700 !important;
  }}

  /* Enhanced Guidelines Box */
  .guidelines-box {{ 
    background: {"linear-gradient(145deg, " + COL['dark'] + " 0%, #2a4a37 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #f1f8e9 0%, #ffffff 100%)"}; 
    padding: 25px; 
    border-radius: 15px; 
    color: {COL['text_primary']}; 
    margin-bottom: 30px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.1);
    border-left: 6px solid {COL['highlight']};
    border: 1px solid {COL['border_color']};
  }}
  
  .guidelines-box h4 {{ 
    color: {COL['highlight']}; 
    margin-top: 0; 
    font-size: 1.4rem; 
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }}
  
  .guidelines-box ul {{ 
    padding-left: 1rem; 
    margin-bottom: 0; 
    list-style-type: none; 
  }}
  
  .guidelines-box ul li {{ 
    padding-left: 2.5rem; 
    position: relative;
    margin-bottom: 12px;
    font-size: 1.15rem;
    line-height: 1.5;
    color: {COL['text_primary']};
  }}
  
  .guidelines-box ul li:before {{ 
    content: "✓"; 
    color: {COL['highlight']};
    position: absolute;
    left: 0;
    font-weight: bold;
    font-size: 1.3rem;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }}
  
  /* Enhanced Results Section */
  .results-header {{
    text-align: center;
    color: {COL['highlight']};
    margin: 40px 0 30px;
    font-size: 2.2rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 4px 8px rgba(0,0,0,0.15);
    background: linear-gradient(45deg, {COL['highlight']}, {COL['text_primary'] if not st.session_state.dark_mode else '#ffffff'});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  
  /* Enhanced Metrics Cards */
  .metric-card {{
    background: linear-gradient(145deg, {COL['gradient_start']} 0%, {COL['gradient_end']} 100%);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15), inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.4s ease;
    margin-bottom: 20px;
    border: 1px solid {COL['border_color']};
    position: relative;
    overflow: hidden;
  }}
  
  .metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    transition: all 0.5s;
  }}
  
  .metric-card:hover {{
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 15px 40px rgba(0,0,0,0.2), 0 0 20px {COL['border_color']};
  }}
  
  .metric-card:hover::before {{
    left: 100%;
  }}
  
  .metric-value {{
    font-size: 2.2rem;
    font-weight: 900;
    margin-bottom: 10px;
    color: white;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }}
  
  .metric-label {{
    font-size: 1.2rem;
    color: rgba(255,255,255,0.9);
    font-weight: 600;
    letter-spacing: 0.5px;
  }}
  
  /* Enhanced Tissue Composition */
  .tissue-item {{
    background: {"linear-gradient(145deg, " + COL['dark'] + " 0%, #2a4a37 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%)"}; 
    padding: 20px 25px;
    margin: 15px 0;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.05);
    border-left: 6px solid transparent;
    transition: all 0.4s ease;
    border: 1px solid {COL['border_color']};
    color: {COL['text_primary']};
  }}
  
  .tissue-item:hover {{
    transform: translateX(8px) scale(1.02);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
  }}
  
  .tissue-name {{
    font-weight: 700;
    font-size: 1.3rem;
    text-transform: capitalize;
    display: flex;
    align-items: center;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    color: {COL['text_primary']};
  }}
  
  .tissue-color-indicator {{
    width: 24px;
    height: 24px;
    border-radius: 6px;
    margin-right: 15px;
    border: 2px solid {"rgba(255,255,255,0.4)" if st.session_state.dark_mode else "rgba(0,0,0,0.2)"};
    display: inline-block;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
  }}
  
  .tissue-stats {{
    display: flex;
    gap: 30px;
    align-items: center;
  }}
  
  .tissue-percent {{
    font-weight: 800;
    font-size: 1.4rem;
    color: {COL['highlight']};
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }}
  
  .tissue-area {{
    font-weight: 600;
    font-size: 1.2rem;
    color: {COL['text_primary']};
    opacity: 0.9;
  }}
  
  /* Enhanced Analysis Tabs */
  .analysis-tab {{
    background: {"linear-gradient(145deg, " + COL['dark'] + " 0%, #2a4a37 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%)"};  
    border-radius: 15px;
    padding: 30px;
    margin: 25px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid {COL['border_color']};
  }}
  
  .tab-title {{
    color: {COL['highlight']};
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 25px;
    text-align: center;
    border-bottom: 3px solid {COL['accent']};
    padding-bottom: 15px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    letter-spacing: 1px;
  }}
  
  /* Enhanced Footer */
  .footer {{ 
    text-align: center; 
    padding: 35px 0; 
    margin-top: 60px; 
    border-top: 3px solid {COL['dark']}; 
    color: {COL['text_primary']}; 
    font-size: 1.2rem; 
    font-weight: 500;
    background: {COL['card_bg']};
    border-radius: 15px 15px 0 0;
  }}
  
  .section-wrapper {{
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 20px 0;
  }}
  
  /* Color legend light mode friendly */
  .color-legend-item {{
    display: flex;
    align-items: center;
    margin: 5px 0;
    color: {COL['text_primary']};
  }}
  
  .color-legend-text {{
    color: {COL['text_primary']};
    font-weight: 600;
    text-transform: capitalize;
  }}
  
  /* Enhanced Responsive Design */
  @media screen and (max-width: 768px) {{
    .header {{ padding: 25px 20px; }}
    .header h1 {{ font-size: 2rem; }}
    .header p {{ font-size: 1.1rem; }}
    .instructions {{ padding: 25px; }}
    .img-container img, .stImage img {{ max-height: 800px !important; }}
    .metric-value {{ font-size: 1.8rem; }}
    .results-header {{ font-size: 1.7rem; }}
    .stButton>button {{ padding: 15px 30px; font-size: 1.1rem; }}
  }}
  
  @media screen and (min-width: 769px) and (max-width: 1024px) {{
    .header h1 {{ font-size: 2.5rem; }}
    .img-container img, .stImage img {{ max-height: 1200px !important; }}
  }}
  
  @media screen and (min-width: 1025px) {{
    .content-wrapper {{ max-width: 1500px; margin: 0 auto; }}
    .section-wrapper {{ max-width: 95%; margin: 0 auto; }}
    .img-container img, .stImage img {{ max-height: 1400px !important; }}
  }}
  
  /* Smooth scrolling */
  html {{
    scroll-behavior: smooth;
  }}
  
  /* Custom scrollbar */
  ::-webkit-scrollbar {{
    width: 12px;
  }}
  
  ::-webkit-scrollbar-track {{
    background: {COL['surface']};
  }}
  
  ::-webkit-scrollbar-thumb {{
    background: linear-gradient(135deg, {COL['accent']}, {COL['highlight']});
    border-radius: 6px;
  }}
  
  ::-webkit-scrollbar-thumb:hover {{
    background: linear-gradient(135deg, {COL['highlight']}, {COL['primary']});
  }}
</style>
""", unsafe_allow_html=True)

# ──── Page Layout ────────────────────────────────────────────────
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# ──── Header ───────────────────────────────────────────────────
if LOGO_PATH.exists():
    st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{base64.b64encode(open(str(LOGO_PATH), 'rb').read()).decode()}" class="logo">
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <h1>🩹 Advanced Wound Analysis</h1>
  <p>Professional AI-Powered Wound Assessment & Tissue Composition Analysis</p>
</div>
""", unsafe_allow_html=True)

# ──── Sugar Heal Model Functions ────────────────────────────────────────────────
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), "float32")
    y_pred_f = K.cast(K.flatten(y_pred), "float32")
    inter    = K.sum(y_true_f * y_pred_f)
    return (2*inter + smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), "float32")
    y_pred_f = K.cast(K.flatten(y_pred), "float32")
    inter    = K.sum(y_true_f * y_pred_f)
    union    = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)

@st.cache_resource
def load_sugar_model():
    with st.spinner("Loading High accuracy segmentation model..."):
        return tf.keras.models.load_model(
            str(SUGAR_MODEL_PATH),
            custom_objects={"dice_coefficient": dice_coefficient, "iou_metric": iou_metric},
            compile=False
        )

def preprocess_sugar(img_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return (img_rgb.astype("float32") / 255)[None, ...]

def predict_wound_mask(img_bgr: np.ndarray, model) -> np.ndarray:
    prob = model.predict(preprocess_sugar(img_bgr), verbose=0)[0, ..., 0]
    mask = (prob > THRESHOLD).astype("uint8") * 255
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask

def make_overlay(orig_bgr, mask):
    h, w = orig_bgr.shape[:2]
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask

    mask_r = cv2.resize(mask_gray, (w, h), cv2.INTER_NEAREST)
    overlay = orig_bgr.copy()
    overlay[mask_r==255] = (122,164,140)
    return cv2.addWeighted(overlay, ALPHA, orig_bgr, 1-ALPHA, 0)

def calculate_wound_area(mask):
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask
    return int(np.sum(mask_gray > 0))

# ──── Tissue Analysis Model Functions ────────────────────────────────────────────
@st.cache_resource
def load_tissue_model():
    with st.spinner("Loading tissue analysis model..."):
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=N_CLASSES,
            decoder_attention_type='scse',
            activation=None,
        )
        state_dict = torch.load(str(TISSUE_MODEL_PATH), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

def preprocess_tissue(image_pil):
    image = np.array(image_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

def postprocess_tissue(mask):
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    mask = mask.argmax(0).cpu().numpy()

    # Create color mask using our centralized color system
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, class_name in enumerate(CLASS_NAMES):
        # Skip unused classes
        if class_name not in DISPLAY_CLASSES:
            continue
        color_bgr = TISSUE_COLORS_BGR[class_name]
        color_mask[mask == idx] = color_bgr

    return color_mask, mask

def calculate_tissue_percentages(mask, class_names):
    total_pixels = mask.size
    percentages = {}
    for idx, name in enumerate(class_names):
        # Skip unused classes
        if name not in DISPLAY_CLASSES:
            continue
        class_pixels = np.sum(mask == idx)
        if class_pixels > 0:
            percentages[name] = (class_pixels / total_pixels) * 100
    return percentages

def calculate_tissue_percentages_and_areas(mask, class_names):
    total_pixels = mask.size
    data = {}
    for idx, name in enumerate(class_names):
        # Skip unused classes
        if name not in DISPLAY_CLASSES:
            continue
        class_pixels = np.sum(mask == idx)
        if class_pixels > 0:
            data[name] = {
                'percentage': (class_pixels / total_pixels) * 100,
                'area_px': int(class_pixels)
            }
    return data

def get_dominant_tissue(tissue_data):
    """Get dominant tissue excluding background"""
    non_background = {k: v['percentage'] for k, v in tissue_data.items() if k != "background" and v['percentage'] > 0}
    if non_background:
        dominant = max(non_background.items(), key=lambda x: x[1])
        return (dominant[0], dominant[1])
    else:
        return ("background", tissue_data.get("background", {}).get('percentage', 0))

def calculate_health_score(tissue_data):
    """Calculate overall wound health score based on tissue composition"""
    score = 0
    total_weight = 0

    for tissue, info in tissue_data.items():
        percentage = info['percentage']
        if tissue in TISSUE_HEALTH_WEIGHTS and percentage > 0:
            weight = TISSUE_HEALTH_WEIGHTS[tissue]
            score += weight * (percentage / 100)
            total_weight += abs(weight) * (percentage / 100)

    # Normalize to 0-100 scale
    if total_weight > 0:
        normalized_score = ((score + total_weight) / (2 * total_weight)) * 100
        return max(0, min(100, normalized_score))
    return 50  # Neutral score if no tissues detected

def generate_recommendations(tissue_data):
    """Generate healing recommendations based on tissue analysis"""
    recommendations = []

    if tissue_data.get("granulation", {}).get('percentage', 0) > 40:
        recommendations.append("✅ Good granulation tissue - wound healing well")

    if tissue_data.get("fibrin", {}).get('percentage', 0) > 20:
        recommendations.append("💧 Maintain moist wound environment")

    return recommendations if recommendations else ["📋 Continue current wound care regimen"]

# ──── Load Models ────────────────────────────────────────────────
try:
    sugar_model = load_sugar_model()
    tissue_model = load_tissue_model()
    st.success("✅ Both AI models loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ──── Instructions ─────────────────────────────────────────────
st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
st.markdown("""
<div class="instructions">
  <strong>🔬 Advanced Wound Analysis System:</strong><br>
  <ol>
    <li><b>Upload</b> a clear wound image (PNG/JPG/JPEG)</li>
    <li><b>Choose</b> analysis mode: Basic segmentation or Complete analysis</li>
    <li><b>Analyze</b> to get precise wound boundaries + detailed tissue composition</li>
    <li><b>View</b> comprehensive results with tissue breakdown and healing recommendations</li>
  </ol>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──── Analysis Mode Selection ────────────────────────────────────────────────
st.markdown(f"""
<style>
/* Make the radio buttons text much larger */
.stRadio > div {{
    flex-direction: column;
}}

.stRadio > div label {{
    font-size: 2rem !important;
    padding: 15px !important;
    margin: 10px 0 !important;
    background: {COL['card_bg']} !important;
    border-radius: 10px !important;
    border-left: 4px solid {COL['highlight']} !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
}}

.stRadio > div label:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
}}

/* Style the label above the radio buttons */
.stRadio > label {{
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: {COL['highlight']} !important;
    margin-bottom: 15px !important;
    text-align: center !important;
    display: block !important;
}}
</style>

<div style="
    text-align: center;
    margin: 30px 0 10px 0;
">
    <h2 style="
        font-size: 2.4rem;
        font-weight: 800;
        color: {COL['highlight']};
        margin: 0;
    ">🎯 Analysis Mode</h2>
</div>
""", unsafe_allow_html=True)

analysis_mode = st.radio(
    "Select analysis type:",
    ["🔍 Basic Segmentation (Fast)", "🧬 Complete Analysis (Detailed)"],
    help="Basic: Wound boundary detection only. Complete: Full tissue analysis + recommendations"
)
# ──── Upload & Analysis ────────────────────────────────────────
col1, col2 = st.columns([2, 1]) 

with col1:
    uploaded = st.file_uploader("Upload wound image", type=["png","jpg","jpeg"])

with col2:
    st.markdown("""
    <div class="guidelines-box">
        <h4>📸 Image Guidelines</h4>
        <ul>
            <li>Good lighting & focus</li>
            <li>Wound clearly visible</li>
            <li>Consistent distance</li>
            <li>Include reference scale</li>
            <li>Clean wound area</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Display uploaded image
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(pil, caption="Uploaded Wound Image", use_container_width=True, 
             output_format="PNG", clamp=True, channels="RGB")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Analysis button
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    if st.button("🚀 Analyze Wound", help="Click to run AI analysis"):

        if "Basic" in analysis_mode:
            # ──── Basic Segmentation Analysis ────────────────────────────────────
            with st.spinner("Running basic wound segmentation..."):
                progress = st.progress(0)
                for i in range(100):
                    progress.progress(i+1)
                    if i==50:
                        wound_mask = predict_wound_mask(orig_bgr, sugar_model)
                        overlay = make_overlay(orig_bgr, wound_mask)
                        area = calculate_wound_area(wound_mask)
                progress.empty()

            st.success("✅ Basic analysis complete!")
            st.markdown('<div class="results-header">Wound Segmentation Results</div>', unsafe_allow_html=True)

            # Display results
            col1, col2 = st.columns(2)

            # Prepare images for display
            if len(wound_mask.shape) == 2:
                display_mask = cv2.cvtColor(wound_mask, cv2.COLOR_GRAY2RGB)
            else:
                display_mask = wound_mask

            overlay_display = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            with col1:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(display_mask, caption="Wound Boundary Mask", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(overlay_display, caption="Wound Overlay (Green)", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)

            # Basic metrics
            st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{area:,}</div>
                    <div class="metric-label">Wound Pixels</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                wound_percentage = (area / (IMG_SIZE * IMG_SIZE)) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{wound_percentage:.1f}%</div>
                    <div class="metric-label">Image Coverage</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">Basic</div>
                    <div class="metric-label">Analysis Mode</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # ──── Complete Analysis (Both Models) ────────────────────────────────────
            with st.spinner("Running complete wound analysis..."):
                progress = st.progress(0)

                # Step 1: Basic segmentation
                for i in range(30):
                    progress.progress(i+1)

                wound_mask = predict_wound_mask(orig_bgr, sugar_model)
                overlay = make_overlay(orig_bgr, wound_mask)
                area = calculate_wound_area(wound_mask)

                # Step 2: Tissue analysis
                for i in range(30, 70):
                    progress.progress(i+1)

                with torch.no_grad():
                    tensor_img = preprocess_tissue(pil)
                    tissue_pred = tissue_model(tensor_img)
                    tissue_mask_bgr, tissue_mask_indices = postprocess_tissue(tissue_pred)
                    tissue_data = calculate_tissue_percentages_and_areas(tissue_mask_indices, CLASS_NAMES)

                # Step 3: Analysis completion
                for i in range(70, 100):
                    progress.progress(i+1)

                health_score = calculate_health_score(tissue_data)
                dominant_tissue, dominant_percent = get_dominant_tissue(tissue_data)
                recommendations = generate_recommendations(tissue_data)

                progress.empty()

            st.success("✅ Complete analysis finished!")
            st.markdown('<div class="results-header">Advanced Wound Analysis Results</div>', unsafe_allow_html=True)

            # ──── Image Results Display ────────────────────────────────────────────────
            st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            # Prepare images for display
            if len(wound_mask.shape) == 2:
                display_mask = cv2.cvtColor(wound_mask, cv2.COLOR_GRAY2RGB)
            else:
                display_mask = wound_mask

            overlay_display = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            tissue_display = cv2.cvtColor(tissue_mask_bgr, cv2.COLOR_BGR2RGB)

            # *** ACTUAL SWAP HERE *** - Left shows tissue, right shows wound boundary
            with col1:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(display_mask, caption="🧬 Tissue Composition Analysis", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="img-container">', unsafe_allow_html=True)
                st.image(tissue_display, caption="🎯 Wound Boundary Detection", 
                         use_container_width=True, clamp=True, output_format="PNG")
                st.markdown('</div>', unsafe_allow_html=True)
   
            st.markdown('</div>', unsafe_allow_html=True)

            # *** SWAPPED Combined overlay - Now shows wound overlay instead of tissue ***
            st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
            st.markdown('<div class="img-container">', unsafe_allow_html=True)

            alpha = 0.5  # or any value you prefer
            orig_bgr_resized = cv2.resize(orig_bgr, (IMG_SIZE, IMG_SIZE))
            tissue_overlay = cv2.addWeighted(orig_bgr_resized, 1 - alpha, tissue_mask_bgr, alpha, 0)
            tissue_overlay_rgb = cv2.cvtColor(tissue_overlay, cv2.COLOR_BGR2RGB)
            st.image(tissue_overlay_rgb, caption="🔗 Combined Analysis Overlay", 
                     use_container_width=True, clamp=True, output_format="PNG")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ──── Key Metrics Dashboard ────────────────────────────────────────────────
            st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{health_score:.0f}</div>
                    <div class="metric-label">Health Score</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{area:,}</div>
                    <div class="metric-label">Wound Area (px)</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{dominant_tissue.title()}</div>
                    <div class="metric-label">Dominant Tissue</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                tissue_count = len([k for k, v in tissue_data.items() if k != "background" and v['percentage'] > 0])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{tissue_count}</div>
                    <div class="metric-label">Tissue Types</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # ──── Detailed Analysis Tabs ────────────────────────────────────────────────
            tab1, tab2, tab3 = st.tabs(["🧬 Tissue Composition", "📊 Health Assessment", "💡 Recommendations"])

            with tab1:
                st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
                st.markdown('<div class="tab-title">Tissue Composition Breakdown</div>', unsafe_allow_html=True)

                # Color legend first
                st.markdown("Color Legend:")
                legend_cols = st.columns(3)
                for i, (tissue, color) in enumerate(TISSUE_COLORS_HEX.items()):
                    # Skip unused classes
                    if tissue not in DISPLAY_CLASSES:
                        continue
                    if tissue in tissue_data and tissue_data[tissue]['percentage'] > 0:
                        col_idx = i % 3
                        with legend_cols[col_idx]:
                            st.markdown(f"""
                            <div class="color-legend-item">
                                <div style="width: 20px; height: 20px; background-color: {color}; 
                                     border-radius: 4px; margin-right: 10px; border: 1px solid {'#fff' if st.session_state.dark_mode else '#000'};"></div>
                                <span class="color-legend-text">{tissue}</span>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("---")

                # Tissue percentages with area
                sorted_tissues = sorted(
                    [(k, v) for k, v in tissue_data.items() if v['percentage'] > 0], 
                    key=lambda x: x[1]['percentage'], reverse=True
                )

                for tissue, info in sorted_tissues:
                    color = TISSUE_COLORS_HEX[tissue]
                    st.markdown(f"""
                    <div class="tissue-item" style="border-left-color: {color};">
                        <div class="tissue-name">
                            <div class="tissue-color-indicator" style="background-color: {color};"></div>
                            {tissue.title()}
                        </div>
                        <div class="tissue-stats">
                            <div class="tissue-percent">{info['percentage']:.1f}%</div>
                            <div class="tissue-area">{info['area_px']:,} px</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
                st.markdown('<div class="tab-title">Health Assessment</div>', unsafe_allow_html=True)

                # Health score interpretation
                if health_score >= 80:
                    health_status = "Excellent"
                    health_color = COL['success']
                    health_icon = "🌟"
                elif health_score >= 60:
                    health_status = "Good"
                    health_color = COL['success']
                    health_icon = "✅"
                elif health_score >= 40:
                    health_status = "Fair"
                    health_color = COL['warning']
                    health_icon = "⚠"
                else:
                    health_status = "Poor"
                    health_color = COL['danger']
                    health_icon = "🚨"

                st.markdown(f"""
                <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {COL['dark']}, {COL['accent']}); 
                     border-radius: 15px; margin: 20px 0; color: white;">
                    <div style="font-size: 4rem; margin-bottom: 10px;">{health_icon}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: {health_color};">{health_score:.0f}/100</div>
                    <div style="font-size: 1.5rem; margin-top: 10px;">Overall Health: {health_status}</div>
                </div>
                """, unsafe_allow_html=True)

                # Detailed breakdown
                st.markdown("Health Score Factors:")

                positive_factors = []
                negative_factors = []

                for tissue, info in tissue_data.items():
                    percentage = info['percentage']
                    if percentage > 1:  # Only show significant tissues
                        weight = TISSUE_HEALTH_WEIGHTS.get(tissue, 0)
                        if weight > 0:
                            positive_factors.append(f"• {tissue.title()}: {percentage:.1f}% (+{weight*100:.0f} points)")
                        elif weight < 0:
                            negative_factors.append(f"• {tissue.title()}: {percentage:.1f}% ({weight*100:.0f} points)")

                if positive_factors:
                    st.markdown("Positive Factors:")
                    for factor in positive_factors:
                        st.markdown(f"<span style='color: {COL['success']};'>{factor}</span>", unsafe_allow_html=True)

                if negative_factors:
                    st.markdown("Concerning Factors:")
                    for factor in negative_factors:
                        st.markdown(f"<span style='color: {COL['danger']};'>{factor}</span>", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
                st.markdown('<div class="tab-title">Clinical Recommendations</div>', unsafe_allow_html=True)

                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div style="background-color: {COL['accent']}; padding: 15px; margin: 10px 0; 
                         border-radius: 10px; border-left: 5px solid {COL['highlight']};">
                        <strong style="color: white; font-size: 1.2rem;">{i}. {rec}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional care guidelines
                st.markdown("General Wound Care Guidelines:")
                guidelines = [
                    "🧼 Keep wound clean and monitor for signs of infection",
                    "💧 Maintain appropriate moisture balance",
                    "🔄 Change dressings as recommended by healthcare provider",
                    "📏 Document wound progress with regular measurements",
                    "👩‍⚕ Consult healthcare provider for concerning changes",
                    "📱 Use this tool for regular monitoring and documentation"
                ]

                for guideline in guidelines:
                    st.markdown(f"""
                    <div style="padding: 8px 0; color: {COL['text_primary']}; font-size: 1.1rem;">
                        {guideline}
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ──── Footer ────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)  # Close content-wrapper

st.markdown("""
<div class="footer">
    <strong>Advanced Wound Analysis System</strong><br>
    Powered by dual AI models for comprehensive wound assessment and monitoring.<br>
    <em>For research and educational purposes. Always consult healthcare professionals for medical decisions.</em>
</div>
""", unsafe_allow_html=True)
