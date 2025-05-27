# streamlit_app.py

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

# Tissue Analysis Config
N_CLASSES = 9
ENCODER = "mit_b3"
CLASS_NAMES = [
    "background", "fibrin", "granulation", "callus", "necrotic", "eschar", "neodermis", "tendon", "dressing"
]

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# ──── THEME SYSTEM ────────────────────────────────────────────────
# Theme definitions with CSS tokens
THEMES = {
    'dark': {
        "primary": "#074225",
        "secondary": "#41706F",
        "accent": "#3B6C53",
        "dark": "#335F4B",
        "light": "#81A295",
        "surface": "#1a1a1a",
        "surface_alt": "#2a2a2a",
        "text_primary": "#E0E0E0",
        "text_secondary": "#FFFFFF",
        "text_muted": "#B0B0B0",
        "highlight": "rgb(122,164,140)",
        "success": "#28a745",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "gradient_start": "#074225",
        "gradient_end": "#3B6C53",
        "card_bg": "linear-gradient(145deg, #335F4B 0%, #2a4a37 100%)",
        "card_border": "rgba(122,164,140,0.2)",
        "overlay_bg": "rgba(7, 66, 37, 0.1)",
        "shadow": "rgba(0,0,0,0.4)",
        "shadow_light": "rgba(0,0,0,0.3)",
    },
    'light': {
        "primary": "#0a5a2e",
        "secondary": "#2d5a58",
        "accent": "#4a7a63",
        "dark": "#1a3d2b",
        "light": "#a5c5b5",
        "surface": "#ffffff",
        "surface_alt": "#f8f9fa",
        "text_primary": "#2c3e50",
        "text_secondary": "#1a1a1a",
        "text_muted": "#6c757d",
        "highlight": "rgb(74,122,99)",
        "success": "#28a745",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "gradient_start": "#0a5a2e",
        "gradient_end": "#4a7a63",
        "card_bg": "linear-gradient(145deg, #ffffff 0%, #f1f3f4 100%)",
        "card_border": "rgba(74,122,99,0.3)",
        "overlay_bg": "rgba(10, 90, 46, 0.1)",
        "shadow": "rgba(0,0,0,0.15)",
        "shadow_light": "rgba(0,0,0,0.1)",
    }
}

# Get current theme colors
COL = THEMES[st.session_state.theme]

# ──── CENTRALIZED COLOR CONTROL ────────────────────────────────────────────────
# Define tissue colors (RGB format for display)
TISSUE_COLORS_RGB = {
    "background": (0, 0, 0),         # Black
    "fibrin": (255, 255, 0),         # Yellow  
    "granulation": (255, 0, 0),      # Red
    "callus": (0, 0, 255),           # Blue
    "necrotic": (255, 165, 0),       # Orange
    "eschar": (128, 0, 128),         # Purple
    "neodermis": (0, 255, 255),      # Cyan
    "tendon": (255, 192, 203),       # Pink
    "dressing": (0, 255, 0),         # Green
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
    "neodermis": 0.9,      # Excellent - new skin
    "fibrin": 0.6,         # Moderate - part of healing
    "callus": 0.4,         # Poor - hard tissue
    "necrotic": -0.8,      # Very bad - dead tissue
    "eschar": -0.6,        # Bad - scab tissue
    "tendon": 0.2,         # Neutral - exposed but not necessarily bad
    "dressing": 0.0,       # Neutral
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

# ──── Theme Toggle Function ────────────────────────────────────────────────
def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# ──── Enhanced CSS with Theme Support ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Base Styles */
  .stApp {{ 
    background: linear-gradient(135deg, {COL['surface']} 0%, {COL['surface_alt']} 100%); 
    color: {COL['text_primary']}; 
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif; 
    transition: all 0.3s ease;
  }}
  
  /* Remove default Streamlit image behavior */
  [data-testid="stImage"] {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
  }}
  
  [data-testid="stImage"] img {{
    max-width: 100% !important;
    height: auto !important;
    margin: 0 auto !important;
    display: block !important;
  }}
  
  /* Theme Toggle Button */
  .theme-toggle {{
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    background: {COL['gradient_start']};
    color: white;
    border: none;
    border-radius: 50px;
    padding: 12px 20px;
    font-size: 1.2rem;
    cursor: pointer;
    box-shadow: 0 4px 12px {COL['shadow']};
    transition: all 0.3s ease;
  }}
  
  .theme-toggle:hover {{
    background: {COL['accent']};
    transform: translateY(-2px);
    box-shadow: 0 6px 16px {COL['shadow']};
  }}
  
  /* Header Styles with enhanced gradient and animations */
  .header {{ 
    text-align: center; 
    padding: 40px 30px; 
    background: linear-gradient(135deg, {COL['gradient_start']} 0%, {COL['gradient_end']} 50%, {COL['dark']} 100%); 
    color: {COL['text_secondary']}; 
    border-radius: 20px; 
    box-shadow: 0 10px 30px {COL['shadow']}, 0 0 50px rgba(122,164,140,0.1); 
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
  }}
  
  /* Enhanced Instructions Box */
  .instructions {{ 
    background: {COL['card_bg']}; 
    padding: 35px; 
    border-left: 8px solid {COL['highlight']}; 
    border-radius: 15px; 
    margin-bottom: 40px; 
    color: {COL['text_secondary']}; 
    box-shadow: 0 8px 25px {COL['shadow']}, inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid {COL['card_border']};
  }}
  
  .instructions strong {{ 
    color: {COL['highlight']}; 
    font-size: 1.4rem; 
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }}
  
  /* Enhanced Logo Container */
  .logo-container {{
    background: linear-gradient(145deg, {COL['highlight']} 0%, {COL['accent']} 100%); 
    padding: 25px; 
    border-radius: 20px; 
    text-align: center; 
    margin-bottom: 35px;
    box-shadow: 0 8px 25px {COL['shadow']}, inset 0 1px 0 rgba(255,255,255,0.2);
    border: 2px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
  }}
  
  .logo-container:hover {{
    transform: translateY(-2px);
    box-shadow: 0 12px 35px {COL['shadow']}, inset 0 1px 0 rgba(255,255,255,0.2);
  }}
  
  img.logo {{ 
    display: block; 
    margin: 0 auto; 
    width: 100%; 
    max-width: 1000px;
    padding: 10px; 
    transition: all 0.3s ease;
    filter: drop-shadow(0 4px 8px {COL['shadow_light']});
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
    box-shadow: 0 6px 20px {COL['shadow']}; 
    width: 100%;
    font-size: 1.3rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    position: relative;
    overflow: hidden;
  }}
  
  .stButton>button:hover {{ 
    background: linear-gradient(135deg, {COL['accent']} 0%, {COL['gradient_start']} 50%, {COL['primary']} 100%); 
    transform: translateY(-4px); 
    box-shadow: 0 10px 30px {COL['shadow']}, 0 0 20px rgba(122,164,140,0.3); 
  }}
  
  /* Enhanced File Uploader */
  .css-1cpxqw2, [data-testid="stFileUploader"] {{ 
    border: 3px dashed {COL['accent']}; 
    background: {COL['overlay_bg']}; 
    border-radius: 20px; 
    padding: 35px; 
    transition: all 0.4s ease;
    backdrop-filter: blur(10px);
  }}
  
  .css-1cpxqw2:hover, [data-testid="stFileUploader"]:hover {{ 
    border-color: {COL['highlight']}; 
    background: {COL['card_bg']};
    box-shadow: 0 8px 25px rgba(122,164,140,0.2);
    transform: translateY(-2px);
  }}
  
  /* Enhanced Image Container with Zoom Support */
  .img-container {{ 
    background: {COL['card_bg']}; 
    padding: 30px; 
    border-radius: 20px; 
    box-shadow: 0 8px 25px {COL['shadow']}, inset 0 1px 0 rgba(255,255,255,0.1); 
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
    border: 1px solid {COL['card_border']};
  }}
  
  .zoomable-container {{
    position: relative;
    width: 100%;
    height: 600px;
    overflow: hidden;
    border-radius: 12px;
    cursor: grab;
    background: {COL['surface']};
    display: flex;
    justify-content: center;
    align-items: center;
  }}
  
  .zoomable-container:active {{
    cursor: grabbing;
  }}
  
  .zoomable-image {{
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    transform-origin: center center;
    transition: transform 0.1s ease;
    border-radius: 8px;
    position: absolute;
  }}
  
  /* Enhanced Guidelines Box */
  .guidelines-box {{ 
    background: {COL['card_bg']}; 
    padding: 25px; 
    border-radius: 15px; 
    color: {COL['text_secondary']}; 
    margin-bottom: 30px;
    box-shadow: 0 6px 20px {COL['shadow']}, inset 0 1px 0 rgba(255,255,255,0.1);
    border-left: 6px solid {COL['highlight']};
    border: 1px solid {COL['card_border']};
  }}
  
  .guidelines-box h4 {{ 
    color: {COL['highlight']}; 
    margin-top: 0; 
    font-size: 1.4rem; 
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }}
  
  /* Enhanced Metrics Cards */
  .metric-card {{
    background: linear-gradient(145deg, {COL['dark']} 0%, {COL['accent']} 100%);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    color: white;
    box-shadow: 0 8px 25px {COL['shadow']}, inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.4s ease;
    margin-bottom: 20px;
    border: 1px solid {COL['card_border']};
  }}
  
  .metric-card:hover {{
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 15px 40px {COL['shadow']}, 0 0 20px rgba(122,164,140,0.2);
  }}
  
  .metric-value {{
    font-size: 2.2rem;
    font-weight: 900;
    margin-bottom: 10px;
    color: {COL['text_secondary']};
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
  }}
  
  /* Enhanced Tissue Composition */
  .tissue-item {{
    background: {COL['card_bg']};
    padding: 20px 25px;
    margin: 15px 0;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 5px 15px {COL['shadow_light']}, inset 0 1px 0 rgba(255,255,255,0.05);
    border-left: 6px solid transparent;
    transition: all 0.4s ease;
    border: 1px solid {COL['card_border']};
  }}
  
  .tissue-item:hover {{
    transform: translateX(8px) scale(1.02);
    box-shadow: 0 8px 25px {COL['shadow']};
  }}
  
  .tissue-name {{
    font-weight: 700;
    font-size: 1.3rem;
    text-transform: capitalize;
    display: flex;
    align-items: center;
    color: {COL['text_primary']};
  }}
  
  .tissue-percent {{
    font-weight: 800;
    font-size: 1.4rem;
    color: {COL['highlight']};
  }}
  
  /* Enhanced Analysis Tabs */
  .analysis-tab {{
    background: {COL['card_bg']};
    border-radius: 15px;
    padding: 30px;
    margin: 25px 0;
    box-shadow: 0 8px 25px {COL['shadow']}, inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid {COL['card_border']};
  }}
  
  .tab-title {{
    color: {COL['highlight']};
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 25px;
    text-align: center;
    border-bottom: 3px solid {COL['accent']};
    padding-bottom: 15px;
  }}
  
  /* Control Panel Styles */
  .control-panel {{
    background: {COL['card_bg']};
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
    box-shadow: 0 4px 12px {COL['shadow_light']};
    border: 1px solid {COL['card_border']};
  }}
  
  .control-title {{
    color: {COL['highlight']};
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 15px;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }}
  
  /* Fix section spacing */
  .section-wrapper {{
    margin-bottom: 20px;
  }}
  
  /* Custom slider styling */
  .stSlider > div > div > div > div {{
    background: {COL['accent']};
  }}
  
  /* Enhanced Footer */
  .footer {{ 
    text-align: center; 
    padding: 35px 0; 
    margin-top: 60px; 
    border-top: 3px solid {COL['dark']}; 
    color: {COL['text_muted']}; 
    font-size: 1.2rem; 
    font-weight: 500;
    background: {COL['overlay_bg']};
    border-radius: 15px 15px 0 0;
  }}
  
  /* Responsive Design */
  @media screen and (max-width: 768px) {{
    .header {{ padding: 25px 20px; }}
    .header h1 {{ font-size: 2rem; }}
    .zoomable-container {{ height: 400px; }}
    .theme-toggle {{ top: 10px; right: 10px; padding: 8px 15px; font-size: 1rem; }}
  }}
</style>
""", unsafe_allow_html=True)

# ──── Theme Toggle in Sidebar ────────────────────────────────────────────────
st.markdown(f"""
<div style="position: fixed; top: 20px; right: 20px; z-index: 1000;">
    <button class="theme-toggle" onclick="document.querySelector('[data-testid=\\"baseButton-secondary\\"]').click()">
        {'🌞' if st.session_state.theme == 'dark' else '🌙'} {'Light' if st.session_state.theme == 'dark' else 'Dark'}
    </button>
</div>
""", unsafe_allow_html=True)

if st.button("Toggle Theme", type="secondary", key="theme_toggle"):
    toggle_theme()
    st.rerun()

# ──── Zoom & Pan JavaScript Component ────────────────────────────────────────────────
def create_zoomable_image(image_data, image_id, caption=""):
    """Create a zoomable and pannable image component"""
    image_b64 = base64.b64encode(image_data).decode()
    
    return st.components.v1.html(f"""
    <div class="img-container">
        <div class="control-panel">
            <div class="control-title">
                <span>🔍</span>
                <span>Image Controls</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <button onclick="zoomIn('{image_id}')" style="background: {COL['accent']}; color: white; border: none; padding: 8px 15px; border-radius: 8px; cursor: pointer; font-weight: 600;">Zoom In</button>
                <button onclick="resetZoom('{image_id}')" style="background: {COL['primary']}; color: white; border: none; padding: 8px 15px; border-radius: 8px; cursor: pointer; font-weight: 600;">Reset</button>
                <button onclick="zoomOut('{image_id}')" style="background: {COL['dark']}; color: white; border: none; padding: 8px 15px; border-radius: 8px; cursor: pointer; font-weight: 600;">Zoom Out</button>
            </div>
            <div style="text-align: center; color: {COL['text_primary']}; font-size: 0.9rem; margin-top: 10px;">
                Click and drag to pan • Use buttons or touch gestures to zoom
            </div>
        </div>
        <div class="zoomable-container" id="container_{image_id}">
            <img id="{image_id}" src="data:image/png;base64,{image_b64}" 
                 class="zoomable-image" alt="{caption}" />
        </div>
        <div style="text-align: center; color: {COL['highlight']}; font-size: 1.2rem; font-weight: 700; margin-top: 15px;">
            {caption}
        </div>
    </div>
    
    <script>
    (function() {{
        const img = document.getElementById('{image_id}');
        const container = document.getElementById('container_{image_id}');
        let scale = 1;
        let isDragging = false;
        let startX, startY, initialX = 0, initialY = 0;
        
        // Center the image initially
        function centerImage() {{
            const containerRect = container.getBoundingClientRect();
            const imgRect = img.getBoundingClientRect();
            initialX = 0;
            initialY = 0;
            img.style.transform = `scale(${{scale}}) translate(${{initialX}}px, ${{initialY}}px)`;
        }}
        
        // Call centerImage when loaded
        img.onload = centerImage;
        window.addEventListener('resize', centerImage);
        
        // Zoom functions
        window.zoomIn = function(id) {{
            if (id === '{image_id}') {{
                scale = Math.min(scale * 1.3, 5);
                img.style.transform = `scale(${{scale}}) translate(${{initialX}}px, ${{initialY}}px)`;
            }}
        }};
        
        window.zoomOut = function(id) {{
            if (id === '{image_id}') {{
                scale = Math.max(scale / 1.3, 0.5);
                img.style.transform = `scale(${{scale}}) translate(${{initialX}}px, ${{initialY}}px)`;
            }}
        }};
        
        window.resetZoom = function(id) {{
            if (id === '{image_id}') {{
                scale = 1;
                initialX = 0;
                initialY = 0;
                img.style.transform = 'scale(1) translate(0px, 0px)';
            }}
        }};
        
        // Pan functionality
        img.addEventListener('mousedown', function(e) {{
            isDragging = true;
            startX = e.clientX - initialX;
            startY = e.clientY - initialY;
            img.style.cursor = 'grabbing';
        }});
        
        document.addEventListener('mousemove', function(e) {{
            if (!isDragging) return;
            e.preventDefault();
            initialX = e.clientX - startX;
            initialY = e.clientY - startY;
            img.style.transform = `scale(${{scale}}) translate(${{initialX}}px, ${{initialY}}px)`;
        }});
        
        document.addEventListener('mouseup', function() {{
            isDragging = false;
            img.style.cursor = 'grab';
        }});
        
        // Touch support for mobile
        let lastTouchDistance = 0;
        
        container.addEventListener('touchstart', function(e) {{
            if (e.touches.length === 2) {{
                lastTouchDistance = Math.hypot(
                    e.touches[0].pageX - e.touches[1].pageX,
                    e.touches[0].pageY - e.touches[1].pageY
                );
            }} else if (e.touches.length === 1) {{
                isDragging = true;
                startX = e.touches[0].clientX - initialX;
                startY = e.touches[0].clientY - initialY;
            }}
        }});
        
        container.addEventListener('touchmove', function(e) {{
            e.preventDefault();
            if (e.touches.length === 2) {{
                const touchDistance = Math.hypot(
                    e.touches[0].pageX - e.touches[1].pageX,
                    e.touches[0].pageY - e.touches[1].pageY
                );
                const delta = touchDistance / lastTouchDistance;
                scale = Math.max(0.5, Math.min(5, scale * delta));
                lastTouchDistance = touchDistance;
                img.style.transform = `scale(${{scale}}) translate(${{initialX}}px, ${{initialY}}px)`;
            }} else if (e.touches.length === 1 && isDragging) {{
                initialX = e.touches[0].clientX - startX;
                initialY = e.touches[0].clientY - startY;
                img.style.transform = `scale(${{scale}}) translate(${{initialX}}px, ${{initialY}}px)`;
            }}
        }});
        
        container.addEventListener('touchend', function() {{
            isDragging = false;
        }});
    }})();
    </script>
    """, height=800)

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

def make_overlay(orig_bgr, mask, opacity=ALPHA):
    h, w = orig_bgr.shape[:2]
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask

    mask_r = cv2.resize(mask_gray, (w, h), cv2.INTER_NEAREST)
    overlay = orig_bgr.copy()
    overlay[mask_r==255] = (122,164,140)
    return cv2.addWeighted(overlay, opacity, orig_bgr, 1-opacity, 0)

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
        color_bgr = TISSUE_COLORS_BGR[class_name]
        color_mask[mask == idx] = color_bgr

    return color_mask, mask

def calculate_tissue_percentages_with_area(mask, class_names):
    total_pixels = mask.size
    percentages = {}
    areas = {}
    
    for idx, name in enumerate(class_names):
        class_pixels = np.sum(mask == idx)
        if class_pixels > 0:
            percentages[name] = (class_pixels / total_pixels) * 100
            areas[name] = int(class_pixels)  # Area in pixels
    
    return percentages, areas

def get_dominant_tissue(tissue_percentages):
    """Get dominant tissue excluding background"""
    non_background = {k: v for k, v in tissue_percentages.items() if k != "background" and v > 0}
    if non_background:
        return max(non_background.items(), key=lambda x: x[1])
    else:
        return ("background", tissue_percentages.get("background", 0))

def calculate_health_score(tissue_percentages):
    """Calculate overall wound health score based on tissue composition"""
    score = 0
    total_weight = 0

    for tissue, percentage in tissue_percentages.items():
        if tissue in TISSUE_HEALTH_WEIGHTS and percentage > 0:
            weight = TISSUE_HEALTH_WEIGHTS[tissue]
            score += weight * (percentage / 100)
            total_weight += abs(weight) * (percentage / 100)

    # Normalize to 0-100 scale
    if total_weight > 0:
        normalized_score = ((score + total_weight) / (2 * total_weight)) * 100
        return max(0, min(100, normalized_score))
    return 50  # Neutral score if no tissues detected

def generate_recommendations(tissue_percentages):
    """Generate healing recommendations based on tissue analysis"""
    recommendations = []

    if tissue_percentages.get("necrotic", 0) > 5:
        recommendations.append("⚠️ Debridement recommended - significant necrotic tissue present")

    if tissue_percentages.get("eschar", 0) > 10:
        recommendations.append("🧹 Consider eschar removal for better healing")

    if tissue_percentages.get("granulation", 0) > 40:
        recommendations.append("✅ Good granulation tissue - wound healing well")

    if tissue_percentages.get("neodermis", 0) > 0:
        recommendations.append("🌟 New skin formation detected - excellent progress")

    if tissue_percentages.get("fibrin", 0) > 20:
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
    <li><b>Adjust</b> overlay opacity and use zoom/pan controls for detailed inspection</li>
    <li><b>Analyze</b> to get precise wound boundaries + detailed tissue composition with area measurements</li>
    <li><b>View</b> comprehensive results with tissue breakdown and healing recommendations</li>
  </ol>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ──── Analysis Mode Selection ────────────────────────────────────────────────
st.markdown("""
<div class="analysis-mode">
  <h3>🎯 Analysis Mode</h3>
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

    # Display uploaded image with zoom controls
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    
    # Convert PIL to bytes for the zoomable component
    img_buffer = io.BytesIO()
    pil.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    create_zoomable_image(img_bytes, "uploaded_image", "📤 Uploaded Wound Image")
    st.markdown('</div>', unsafe_allow_html=True)

    # Overlay opacity control
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    st.markdown("""
    <div class="control-panel">
        <div class="control-title">
            <span>🎛️</span>
            <span>Overlay Controls</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    overlay_opacity = st.slider(
        "Overlay Opacity",
        min_value=0.0,
        max_value=1.0,
        value=ALPHA,
        step=0.1,
        help="Adjust transparency of analysis overlay"
    )
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
                        overlay = make_overlay(orig_bgr, wound_mask, overlay_opacity)
                        area = calculate_wound_area(wound_mask)
                progress.empty()

            st.success("✅ Basic analysis complete!")
            st.markdown('<div class="results-header">Wound Segmentation Results</div>', unsafe_allow_html=True)

            # Display results with zoom controls
            col1, col2 = st.columns(2)

            # Prepare images for display
            if len(wound_mask.shape) == 2:
                display_mask = cv2.cvtColor(wound_mask, cv2.COLOR_GRAY2RGB)
            else:
                display_mask = wound_mask

            overlay_display = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            with col1:
                # Convert to bytes for zoomable display
                mask_pil = Image.fromarray(display_mask)
                mask_buffer = io.BytesIO()
                mask_pil.save(mask_buffer, format='PNG')
                mask_bytes = mask_buffer.getvalue()
                create_zoomable_image(mask_bytes, "wound_mask", "🎯 Wound Boundary Mask")

            with col2:
                # Convert to bytes for zoomable display
                overlay_pil = Image.fromarray(overlay_display)
                overlay_buffer = io.BytesIO()
                overlay_pil.save(overlay_buffer, format='PNG')
                overlay_bytes = overlay_buffer.getvalue()
                create_zoomable_image(overlay_bytes, "wound_overlay", "🔗 Wound Overlay (Green)")

            # Basic metrics
            st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{area:,}</div>
                    <div class="metric-label">Wound Area (px)</div>
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
                overlay = make_overlay(orig_bgr, wound_mask, overlay_opacity)
                area = calculate_wound_area(wound_mask)

                # Step 2: Tissue analysis
                for i in range(30, 70):
                    progress.progress(i+1)

                with torch.no_grad():
                    tensor_img = preprocess_tissue(pil)
                    tissue_pred = tissue_model(tensor_img)
                    tissue_mask_bgr, tissue_mask_indices = postprocess_tissue(tissue_pred)
                    tissue_percentages, tissue_areas = calculate_tissue_percentages_with_area(tissue_mask_indices, CLASS_NAMES)

                # Step 3: Analysis completion
                for i in range(70, 100):
                    progress.progress(i+1)

                health_score = calculate_health_score(tissue_percentages)
                dominant_tissue, dominant_percent = get_dominant_tissue(tissue_percentages)
                recommendations = generate_recommendations(tissue_percentages)

                progress.empty()

            st.success("✅ Complete analysis finished!")
            st.markdown('<div class="results-header">Advanced Wound Analysis Results</div>', unsafe_allow_html=True)

            # ──── Image Results Display with Zoom Controls ────────────────────────────────────────────────
            st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            # Prepare images for display
            if len(wound_mask.shape) == 2:
                display_mask = cv2.cvtColor(wound_mask, cv2.COLOR_GRAY2RGB)
            else:
                display_mask = wound_mask

            overlay_display = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            tissue_display = cv2.cvtColor(tissue_mask_bgr, cv2.COLOR_BGR2RGB)

            with col1:
                # Wound boundary with zoom
                mask_pil = Image.fromarray(display_mask)
                mask_buffer = io.BytesIO()
                mask_pil.save(mask_buffer, format='PNG')
                mask_bytes = mask_buffer.getvalue()
                create_zoomable_image(mask_bytes, "wound_boundary", "🎯 Wound Boundary Detection")

            with col2:
                # Tissue composition with zoom
                tissue_pil = Image.fromarray(tissue_display)
                tissue_buffer = io.BytesIO()
                tissue_pil.save(tissue_buffer, format='PNG')
                tissue_bytes = tissue_buffer.getvalue()
                create_zoomable_image(tissue_bytes, "tissue_composition", "🧬 Tissue Composition Analysis")

            st.markdown('</div>', unsafe_allow_html=True)

            # Combined overlay with zoom controls
            st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
            
            orig_bgr_resized = cv2.resize(orig_bgr, (IMG_SIZE, IMG_SIZE))
            tissue_overlay = cv2.addWeighted(orig_bgr_resized, 1 - overlay_opacity, tissue_mask_bgr, overlay_opacity, 0)
            tissue_overlay_rgb = cv2.cvtColor(tissue_overlay, cv2.COLOR_BGR2RGB)
            
            combined_pil = Image.fromarray(tissue_overlay_rgb)
            combined_buffer = io.BytesIO()
            combined_pil.save(combined_buffer, format='PNG')
            combined_bytes = combined_buffer.getvalue()
            create_zoomable_image(combined_bytes, "combined_overlay", "🔗 Combined Analysis Overlay")
            
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
                tissue_count = len([k for k, v in tissue_percentages.items() if k != "background" and v > 0])
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
                st.markdown("**Color Legend:**")
                legend_cols = st.columns(3)
                for i, (tissue, color) in enumerate(TISSUE_COLORS_HEX.items()):
                    if tissue in tissue_percentages and tissue_percentages[tissue] > 0:
                        col_idx = i % 3
                        with legend_cols[col_idx]:
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; margin: 5px 0;">
                                <div style="width: 20px; height: 20px; background-color: {color}; 
                                     border-radius: 4px; margin-right: 10px; border: 1px solid #fff;"></div>
                                <span style="color: {COL['text_primary']}; font-weight: 600; text-transform: capitalize;">{tissue}</span>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown("---")

                # Tissue percentages with area information
                sorted_tissues = sorted(
                    [(k, v) for k, v in tissue_percentages.items() if v > 0], 
                    key=lambda x: x[1], reverse=True
                )

                for tissue, percentage in sorted_tissues:
                    area_px = tissue_areas.get(tissue, 0)
                    color = TISSUE_COLORS_HEX[tissue]
                    st.markdown(f"""
                    <div class="tissue-item" style="border-left-color: {color};">
                        <div class="tissue-name">
                            <div class="tissue-color-indicator" style="background-color: {color};"></div>
                            <div>
                                <div style="font-size: 1.3rem; font-weight: 700;">{tissue.title()}</div>
                                <div style="font-size: 0.9rem; color: {COL['text_muted']}; margin-top: 2px;">Area: {area_px:,} px</div>
                            </div>
                        </div>
                        <div class="tissue-percent">{percentage:.1f}%</div>
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
                    health_icon = "⚠️"
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
                st.markdown("**Health Score Factors:**")

                positive_factors = []
                negative_factors = []

                for tissue, percentage in tissue_percentages.items():
                    if percentage > 1:  # Only show significant tissues
                        weight = TISSUE_HEALTH_WEIGHTS.get(tissue, 0)
                        area_px = tissue_areas.get(tissue, 0)
                        if weight > 0:
                            positive_factors.append(f"• {tissue.title()}: {percentage:.1f}% ({area_px:,} px) (+{weight*100:.0f} points)")
                        elif weight < 0:
                            negative_factors.append(f"• {tissue.title()}: {percentage:.1f}% ({area_px:,} px) ({weight*100:.0f} points)")

                if positive_factors:
                    st.markdown("**Positive Factors:**")
                    for factor in positive_factors:
                        st.markdown(f"<span style='color: {COL['success']};'>{factor}</span>", unsafe_allow_html=True)

                if negative_factors:
                    st.markdown("**Concerning Factors:**")
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
                        <strong style="color: {COL['text_secondary']}; font-size: 1.2rem;">{i}. {rec}</strong>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional care guidelines
                st.markdown("**General Wound Care Guidelines:**")
                guidelines = [
                    "🧼 Keep wound clean and monitor for signs of infection",
                    "💧 Maintain appropriate moisture balance",
                    "🔄 Change dressings as recommended by healthcare provider",
                    "📏 Document wound progress with regular measurements",
                    "👩‍⚕️ Consult healthcare provider for concerning changes",
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
    <strong>🩹 Advanced Wound Analysis System</strong><br>
    Powered by dual AI models for comprehensive wound assessment and monitoring.<br>
    <em>For research and educational purposes. Always consult healthcare professionals for medical decisions.</em>
</div>
""", unsafe_allow_html=True)
