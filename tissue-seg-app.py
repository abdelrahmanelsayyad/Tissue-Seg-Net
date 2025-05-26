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

# Try to import segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
except ImportError:
    st.error("segmentation_models_pytorch not found. Please install with: pip install segmentation-models-pytorch")
    st.stop()

st.set_page_config(
    page_title="Sugar Heal – Advanced Wound Analysis",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──── Configuration ─────────────────────────────────────────────
# Binary Segmentation Model (Sugar Heal)
BINARY_MODEL_PATH = Path("unet_wound_segmentation_best.h5")
BINARY_MODEL_URL = "https://drive.google.com/uc?id=1_PToBgQjEKAQAZ9ZX10sRpdgxQ18C-18"

# Tissue Classification Model (PyTorch)
TISSUE_MODEL_PATH = Path("best_model_streamlit.pth")
TISSUE_MODEL_URL = "https://drive.google.com/uc?id=1q0xk9wll0eyF3-CKEc5s6MfG0gE_jde1"

# Tissue Classification Config
N_CLASSES = 9
ENCODER = "mit_b3"
INPUT_SIZE = 256
CLASS_NAMES = [
    "background",    # 0
    "granulation",   # 1
    "callus",        # 2
    "fibrin",        # 3
    "necrotic",      # 4
    "eschar",        # 5
    "neodermis",     # 6
    "tendon",        # 7
    "dressing"       # 8
]
PALETTE = [
    (0, 0, 0),         # 0: background
    (255, 0, 0),       # 1: granulation
    (255, 255, 0),     # 2: callus
    (0, 255, 0),       # 3: fibrin
    (255, 165, 0),     # 4: necrotic
    (128, 0, 128),     # 5: eschar
    (0, 255, 255),     # 6: neodermis
    (255, 192, 203),   # 7: tendon
    (0, 0, 255),       # 8: dressing
]

# General Config
LOGO_PATH = Path("GREEN.png")
THRESHOLD = 0.5
ALPHA = 0.4

# ──── Model Download Functions ──────────────────────────────────
def download_model_if_needed(model_path, model_url, model_name):
    """Download a specific model if it doesn't exist"""
    if not model_path.exists():
        st.info(f"Downloading {model_name}...")
        try:
            with st.spinner(f"Downloading {model_name}..."):
                gdown.download(model_url, str(model_path), quiet=False)
            st.success(f"✅ {model_name} downloaded successfully")
            return True
        except Exception as e:
            st.error(f"❌ Failed to download {model_name}: {e}")
            return False
    return True

# ──── Color Palette & CSS ────────────────────────────────────────
COL = {
    "primary"    : "#074225",
    "secondary"  : "#41706F",
    "accent"     : "#3B6C53",
    "dark"       : "#335F4B",
    "light"      : "#81A295",
    "surface"    : "#202020",
    "text_dark"  : "#E0E0E0",
    "text_light" : "#FFFFFF",
    "highlight"  : "rgb(122,164,140)",
}

# Enhanced CSS
st.markdown(f"""
<style>
  /* Base Styles */
  body {{ background-color: {COL['surface']}; color: {COL['text_dark']}; font-family: 'Helvetica Neue', Arial, sans-serif; }}
  
  /* Header Styles */
  .header {{ 
    text-align: center; 
    padding: 20px; 
    background: linear-gradient(135deg, {COL['primary']}, {COL['dark']}); 
    color: {COL['text_light']}; 
    border-radius: 12px; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.3); 
    margin-bottom: 25px; 
    transition: all 0.3s ease;
  }}
  .header h1 {{ margin:0; font-size:2.2rem; font-weight:600; letter-spacing:1px; }}
  .header p {{ font-size: 1.1rem; margin-top: 8px; opacity: 0.9; }}
  
  /* Instructions Box */
  .instructions {{ 
    background-color: {COL['dark']}; 
    padding: 20px; 
    border-left: 6px solid {COL['accent']}; 
    border-radius: 8px; 
    margin-bottom: 25px; 
    color: {COL['text_light']}; 
    box-shadow: 0 3px 8px rgba(0,0,0,0.2);
  }}
  .instructions strong {{ color:{COL['highlight']}; font-size:1.2rem; }}
  .instructions ol {{ padding-left: 25px; margin-top: 10px; }}
  .instructions li {{ margin-bottom: 5px; }}
  
  /* Logo Container */
  .logo-container {{
    background-color: {COL['highlight']}; 
    padding: 15px; 
    border-radius: 10px; 
    text-align: center; 
    margin-bottom: 20px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.2);
  }}
  img.logo {{ 
    display: block; 
    margin: 0 auto; 
    width: 100%; 
    max-width: 800px;
    padding: 5px; 
    transition: all 0.3s ease;
  }}
  
  /* Button Styling */
  .stButton>button {{ 
    background: linear-gradient(135deg, {COL['primary']}, {COL['dark']}); 
    color: white; 
    border: none; 
    border-radius: 8px; 
    padding: 12px 28px; 
    font-weight: 500; 
    transition: all .3s ease; 
    box-shadow: 0 3px 8px rgba(0,0,0,0.25); 
    width: 100%;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
  }}
  .stButton>button:hover {{ 
    background: linear-gradient(135deg, {COL['accent']}, {COL['primary']}); 
    transform: translateY(-2px); 
    box-shadow: 0 5px 12px rgba(0,0,0,0.35); 
  }}
  
  /* File Uploader */
  .css-1cpxqw2, [data-testid="stFileUploader"] {{ 
    border: 2px dashed {COL['accent']}; 
    background-color: rgba(59, 108, 83, 0.1); 
    border-radius: 10px; 
    padding: 20px; 
    transition: all 0.3s ease;
  }}
  .css-1cpxqw2:hover, [data-testid="stFileUploader"]:hover {{ 
    border-color: {COL['highlight']}; 
    background-color: rgba(59, 108, 83, 0.2);
  }}
  
  /* Image Container */
  .img-container {{ 
    background-color: {COL['dark']}; 
    padding: 20px; 
    border-radius: 12px; 
    box-shadow: 0 4px 10px rgba(0,0,0,0.3); 
    margin-bottom: 20px; 
    text-align: center;
  }}
  
  .img-container img, .stImage img {{ 
    max-height: 450px !important;
    max-width: 100% !important;
    width: auto !important; 
    height: auto !important;
    margin: 0 auto !important; 
    display: block !important; 
    border-radius: 6px !important;
    object-fit: contain !important;
  }}
  
  /* Image Captions */
  .img-container figcaption, .stImage figcaption, .css-1b0udgb, .css-83jbox {{
    font-size: 1.1rem !important;
    color: {COL['text_light']} !important;
    margin-top: 12px !important;
    font-weight: 500 !important;
    text-align: center !important;
  }}
  
  figcaption p {{
    font-size: 1.1rem !important;
    margin: 8px 0 !important;
    color: {COL['text_light']} !important;
    text-align: center !important;
  }}
  
  /* Guidelines Box */
  .guidelines-box {{ 
    background-color: {COL['dark']}; 
    padding: 18px; 
    border-radius: 10px; 
    color: {COL['text_light']}; 
    margin-bottom: 20px;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    border-left: 4px solid {COL['highlight']};
  }}
  .guidelines-box h4 {{ 
    color: {COL['highlight']}; 
    margin-top: 0; 
    font-size: 1.2rem; 
    font-weight: 500;
  }}
  .guidelines-box ul {{ padding-left: .5rem; margin-bottom: 0; list-style-type: none; }}
  .guidelines-box ul li {{ 
    padding-left: 1.5rem; 
    position: relative;
    margin-bottom: 8px;
  }}
  .guidelines-box ul li:before {{ 
    content: "✓"; 
    color: {COL['highlight']};
    position: absolute;
    left: 0;
    font-weight: bold;
  }}
  
  /* Results Section */
  .results-header {{
    text-align: center;
    color: {COL['highlight']};
    margin: 25px 0 15px;
    font-size: 1.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
  }}
  
  /* Metrics Cards */
  .metric-card {{
    background: linear-gradient(135deg, {COL['dark']}, {COL['accent']});
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    margin-bottom: 10px;
  }}
  .metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 5px;
    color: {COL['text_light']};
  }}
  .metric-label {{
    font-size: 1rem;
    color: rgba(255,255,255,0.8);
    font-weight: 500;
  }}
  
  /* Tissue Composition */
  .tissue-card {{
    background: linear-gradient(135deg, {COL['dark']}, {COL['secondary']});
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 8px;
    color: white;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .tissue-name {{
    font-weight: 600;
    text-transform: capitalize;
  }}
  .tissue-percent {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {COL['highlight']};
  }}
  
  /* Footer */
  .footer {{ 
    text-align: center; 
    padding: 20px 0; 
    margin-top: 40px; 
    border-top: 1px solid {COL['dark']}; 
    color: {COL['light']}; 
    font-size: 1rem; 
  }}
  
  /* Responsive */
  @media screen and (max-width: 768px) {{
    .header {{ padding: 15px; }}
    .header h1 {{ font-size: 1.5rem; }}
    .header p {{ font-size: 0.9rem; }}
    .img-container img, .stImage img {{ max-height: 300px !important; }}
    .metric-value {{ font-size: 1.4rem; }}
    .results-header {{ font-size: 1.4rem; }}
  }}
</style>
""", unsafe_allow_html=True)

# ──── Model Loading ────────────────────────────────────────────

# Binary segmentation metrics
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), "float32")
    y_pred_f = K.cast(K.flatten(y_pred), "float32")
    inter = K.sum(y_true_f * y_pred_f)
    return (2*inter + smooth) / (K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def iou_metric(y_true, y_pred, smooth=1):
    y_true_f = K.cast(K.flatten(y_true), "float32")
    y_pred_f = K.cast(K.flatten(y_pred), "float32")
    inter = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)

def load_binary_model():
    """Load the binary wound segmentation model"""
    if not download_model_if_needed(BINARY_MODEL_PATH, BINARY_MODEL_URL, "Binary Segmentation Model"):
        st.stop()
    
    try:
        with st.spinner("Loading binary segmentation model..."):
            model = tf.keras.models.load_model(
                str(BINARY_MODEL_PATH),
                custom_objects={"dice_coefficient": dice_coefficient, "iou_metric": iou_metric},
                compile=False
            )
        return model
    except Exception as e:
        st.error(f"❌ Failed to load binary model: {e}")
        st.stop()

def load_tissue_model():
    """Load the tissue classification model"""
    if not download_model_if_needed(TISSUE_MODEL_PATH, TISSUE_MODEL_URL, "Tissue Classification Model"):
        st.stop()
    
    try:
        with st.spinner("Loading tissue classification model..."):
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
    except Exception as e:
        st.error(f"❌ Failed to load tissue model: {e}")
        st.stop()

# Initialize session state for models
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.binary_model = None
    st.session_state.tissue_model = None

# ──── Processing Functions ─────────────────────────────────────

def preprocess_for_binary(img_bgr: np.ndarray) -> np.ndarray:
    """Preprocess image for binary segmentation"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    return (img_rgb.astype("float32") / 255)[None, ...]

def predict_wound_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Predict binary wound mask"""
    prob = st.session_state.binary_model.predict(preprocess_for_binary(img_bgr), verbose=0)[0, ..., 0]
    mask = (prob > THRESHOLD).astype("uint8")
    return mask

def preprocess_for_tissue(img_pil: Image.Image) -> torch.Tensor:
    """Preprocess image for tissue classification"""
    img = np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE))) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def postprocess_tissue_mask(mask_tensor):
    """Convert tissue prediction to color mask and class map"""
    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor.squeeze(0)
    class_map = mask_tensor.argmax(0).cpu().numpy()  # shape: (H, W)
    color_mask = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        color_mask[class_map == idx] = color
    return color_mask, class_map

def compute_tissue_percentages(class_map, wound_mask):
    """Calculate tissue type percentages within wound area"""
    percentages = {}
    total_wound_pixels = np.sum(wound_mask)
    
    if total_wound_pixels == 0:
        return {name: 0.0 for name in CLASS_NAMES}
    
    for idx, name in enumerate(CLASS_NAMES):
        tissue_pixels = np.sum((class_map == idx) & (wound_mask == 1))
        percent = 100 * tissue_pixels / total_wound_pixels
        if percent > 0.1:  # Only show if >0.1%
            percentages[name] = percent
    
    return percentages

def make_overlay(orig_bgr, mask, color=(122, 164, 140)):
    """Create overlay of mask on original image"""
    h, w = orig_bgr.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), cv2.INTER_NEAREST)
    overlay = orig_bgr.copy()
    overlay[mask_resized == 1] = color
    return cv2.addWeighted(overlay, ALPHA, orig_bgr, 1-ALPHA, 0)

def calculate_wound_area(mask):
    """Calculate wound area in pixels"""
    return int(np.sum(mask > 0))

# ──── Page Layout ────────────────────────────────────────────────

# Header
if LOGO_PATH.exists():
    try:
        st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{base64.b64encode(open(str(LOGO_PATH), 'rb').read()).decode()}" class="logo">
        </div>
        """, unsafe_allow_html=True)
    except:
        pass  # Skip logo if file doesn't exist

st.markdown("""
<div class="header">
  <h1>Sugar Heal – Advanced Wound Analysis</h1>
  <p>AI-powered binary segmentation + tissue classification for comprehensive wound assessment</p>
</div>
""", unsafe_allow_html=True)

# Instructions
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="instructions">
      <strong>📋 How to use this advanced analysis:</strong><br>
      <ol>
        <li>Upload a clear wound image (PNG/JPG/JPEG)</li>
        <li>Click <b>Analyze Wound</b> to run both models</li>
        <li>View binary segmentation results</li>
        <li>Review tissue classification within wound area</li>
        <li>Analyze detailed tissue composition metrics</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="guidelines-box">
        <h4>📸 Image Guidelines</h4>
        <ul>
            <li>Good lighting conditions</li>
            <li>Wound clearly visible</li>
            <li>Consistent distance/scale</li>
            <li>Minimal shadows</li>
            <li>Include reference if possible</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# File Upload
uploaded = st.file_uploader("Upload wound image", type=["png", "jpg", "jpeg"])

if uploaded:
    # Process uploaded image
    pil_image = Image.open(uploaded).convert("RGB")
    orig_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Display original image
    st.markdown('<div class="img-container">', unsafe_allow_html=True)
    st.image(pil_image, caption="Uploaded Wound Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button
    if st.button("🔬 Analyze Wound", help="Click to run comprehensive AI analysis"):
        
        # Load models if not already loaded
        if not st.session_state.models_loaded:
            st.info("Loading AI models for the first time...")
            st.session_state.binary_model = load_binary_model()
            st.session_state.tissue_model = load_tissue_model()
            st.session_state.models_loaded = True
            st.success("✅ All models loaded successfully")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Binary Segmentation
        status_text.text("Step 1/4: Running binary wound segmentation...")
        progress_bar.progress(25)
        
        wound_mask = predict_wound_mask(orig_bgr)
        wound_area = calculate_wound_area(wound_mask)
        
        # Step 2: Create binary overlay
        status_text.text("Step 2/4: Creating segmentation overlay...")
        progress_bar.progress(50)
        
        binary_overlay = make_overlay(orig_bgr, wound_mask)
        binary_overlay_rgb = cv2.cvtColor(binary_overlay, cv2.COLOR_BGR2RGB)
        
        # Step 3: Tissue Classification
        status_text.text("Step 3/4: Running tissue classification...")
        progress_bar.progress(75)
        
        # Apply wound mask to image for tissue classification
        img_masked = np.array(pil_image.resize((INPUT_SIZE, INPUT_SIZE)))
        img_masked[wound_mask == 0] = 0  # Zero out non-wound areas
        
        # Run tissue classification
        input_tensor = preprocess_for_tissue(Image.fromarray(img_masked))
        with torch.no_grad():
            tissue_output = st.session_state.tissue_model(input_tensor)
        
        tissue_color_mask, class_map = postprocess_tissue_mask(tissue_output)
        
        # Step 4: Calculate metrics
        status_text.text("Step 4/4: Computing tissue composition...")
        progress_bar.progress(100)
        
        tissue_percentages = compute_tissue_percentages(class_map, wound_mask)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Comprehensive analysis complete!")
        
        # Display Results
        st.markdown('<div class="results-header">Analysis Results</div>', unsafe_allow_html=True)
        
        # Binary Segmentation Results
        st.markdown("### 🎯 Binary Wound Segmentation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            # Display binary mask
            wound_mask_display = np.stack([wound_mask*255]*3, axis=-1)
            st.image(wound_mask_display, caption="Binary Wound Mask", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(binary_overlay_rgb, caption="Binary Segmentation Overlay", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Binary Segmentation Metrics
        st.markdown("### 📊 Wound Area Metrics")
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{wound_area:,}</div>
                <div class="metric-label">Wound Area (pixels)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            total_pixels = wound_mask.shape[0] * wound_mask.shape[1]
            coverage_pct = wound_area / total_pixels * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{coverage_pct:.2f}%</div>
                <div class="metric-label">Image Coverage</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Tissue Classification Results
        st.markdown("### 🧬 Tissue Classification Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(tissue_color_mask, caption="Tissue Classification Map", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Create tissue overlay on original
            orig_resized = cv2.resize(np.array(pil_image), (INPUT_SIZE, INPUT_SIZE))
            tissue_overlay = (0.6 * orig_resized + 0.4 * tissue_color_mask).astype(np.uint8)
            
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(tissue_overlay, caption="Tissue Classification Overlay", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tissue Composition
        st.markdown("### 🔬 Wound Tissue Composition")
        
        if tissue_percentages:
            # Sort by percentage (descending)
            sorted_tissues = sorted(tissue_percentages.items(), key=lambda x: -x[1])
            
            for tissue_name, percentage in sorted_tissues:
                if tissue_name != "background":  # Skip background
                    st.markdown(f"""
                    <div class="tissue-card">
                        <span class="tissue-name">{tissue_name}</span>
                        <span class="tissue-percent">{percentage:.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No significant tissue types detected within the wound area.")
        
        # Additional Analysis Summary
        st.markdown("### 📈 Analysis Summary")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            dominant_tissue = max(tissue_percentages.items(), key=lambda x: x[1]) if tissue_percentages else ("None", 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{dominant_tissue[0].title()}</div>
                <div class="metric-label">Dominant Tissue</div>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_col2:
            num_tissue_types = len([t for t in tissue_percentages if tissue_percentages[t] > 1.0])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{num_tissue_types}</div>
                <div class="metric-label">Tissue Types (>1%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with summary_col3:
            if tissue_percentages:
                diversity_score = len(tissue_percentages) / len(CLASS_NAMES) * 100
            else:
                diversity_score = 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{diversity_score:.1f}%</div>
                <div class="metric-label">Tissue Diversity</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown(f"<div class='footer'>© 2025 Sugar Heal AI • Advanced Wound Analysis with Binary Segmentation + Tissue Classification</div>", unsafe_allow_html=True)
