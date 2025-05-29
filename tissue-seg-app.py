#latest_version_with_gemini
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
import torch
import gdown
import segmentation_models_pytorch as smp
from fastai.learner import load_learner
import pickle
from fastai.basics import *
import gc

# Gemini AI Integration
import google.generativeai as genai

st.set_page_config(
    page_title="Advanced Wound Analysis",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Create session state variables for models - removed since we're using @st.cache_resource
# Session state now only tracks theme
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# ──── Gemini AI Configuration ────────────────────────────────────────────────
@st.cache_resource
def initialize_gemini():
    """Initialize Gemini AI with API key"""
    try:
        # Configure Gemini API
        os.environ["GEMINI_API_KEY"] = "AIzaSyA21AIdWr6F0UqlR4FwnIf6r3kinLjHe9Q"
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            safety_settings=safety_settings,
            generation_config=generation_config,
            system_instruction="You are a professional medical AI assistant specializing in wound care and analysis. Provide detailed, accurate, and clinically relevant information while emphasizing the importance of professional medical consultation."
        )
        
        return model
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {str(e)}")
        return None

# ──── Config ───────────────────────────────────────────────────

# Advanced Tissue Analysis Model
TISSUE_MODEL_PATH = Path("best_model_streamlit.pth")
TISSUE_MODEL_ID = "1q0xk9wll0eyF3-CKEc5s6MfG0gE_jde1"

# Wound Classification Model
CLASSIFICATION_MODEL_PATH = Path("model.pkl")
CLASSIFICATION_MODEL_ID = "1Itf9SgEjtJwv-7AjY0mWYceDnSX4qIqY"
CLASSIFICATION_MODEL_URL = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"

LOGO_PATH  = Path("GREEN.png")
IMG_SIZE   = 256
ALPHA      = 0.4
MAX_IMAGE_SIZE = 1024  # Maximum dimension for uploaded images

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

# Memory optimization functions
def resize_image_if_needed(image, max_size=MAX_IMAGE_SIZE):
    """Resize image if it exceeds maximum size to reduce memory usage"""
    height, width = image.shape[:2]
    
    if max(height, width) > max_size:
        # Calculate scaling factor
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return image

def clear_memory():
    """Clear memory by garbage collecting"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Download models from Google Drive if not present
def download_models():
    download_success = True

    if not TISSUE_MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        st.info("Downloading tissue analysis model...")
        try:
            gdown.download(f"https://drive.google.com/uc?id={TISSUE_MODEL_ID}", str(TISSUE_MODEL_PATH), quiet=False)
            if not TISSUE_MODEL_PATH.exists():
                st.error(f"Failed to download tissue analysis model to {TISSUE_MODEL_PATH}")
                download_success = False
        except Exception as e:
            st.error(f"Error downloading tissue analysis model: {str(e)}")
            download_success = False
        
    if not CLASSIFICATION_MODEL_PATH.exists():
        try:
            import gdown
        except ImportError:
            os.system(f"{sys.executable} -m pip install gdown")
            import gdown
        st.info("Downloading wound classification model...")
        try:
            gdown.download(CLASSIFICATION_MODEL_URL, str(CLASSIFICATION_MODEL_PATH), quiet=False)
            if not CLASSIFICATION_MODEL_PATH.exists():
                st.error(f"Failed to download classification model to {CLASSIFICATION_MODEL_PATH}")
                download_success = False
        except Exception as e:
            st.error(f"Error downloading classification model: {str(e)}")
            download_success = False
    
    return download_success

# Ensure models are available
model_download_success = download_models()

# ──── Gemini AI Functions ────────────────────────────────────────────────

def generate_health_assessment(tissue_data, wound_type, confidence):
    """Generate detailed health assessment using Gemini AI"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return "Health assessment unavailable - AI service not available"
        
        prompt = f"""
        As a wound care specialist, provide a detailed health assessment for this wound analysis:

        Tissue Composition:
        {format_tissue_data_for_prompt(tissue_data)}
        
        Wound Classification: {wound_type} (Confidence: {confidence:.1%})
        
        Please provide:
        1. Overall wound health status interpretation
        2. Healing stage assessment
        3. Risk factors identified
        4. Prognostic indicators
        5. Key monitoring parameters
        
        Keep it professional and clinical, but accessible to healthcare providers.
        
        CRITICAL: Do not use ANY markdown formatting including asterisks (*), double asterisks (**), 
        underscores (_), hash symbols (#), or any other markdown syntax. 
        Use only plain text with clear formatting and proper paragraphs.
        """
        
        chat = gemini_model.start_chat()
        response = chat.send_message(prompt)
        
        # Clean up any markdown formatting aggressively
        cleaned_response = clean_markdown_formatting(response.text)
        
        return cleaned_response
        
    except Exception as e:
        return f"Health assessment unavailable: {str(e)}"

def generate_wound_classification_info(wound_type, confidence, tissue_data):
    """Generate detailed wound classification information using Gemini AI"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return "Classification information unavailable - AI service not available"
        
        prompt = f"""
        Provide comprehensive information about {wound_type} wounds:

        Current Analysis:
        - Wound Type: {wound_type}
        - AI Confidence: {confidence:.1%}
        - Tissue Composition: {format_tissue_data_for_prompt(tissue_data)}
        
        Please provide detailed information about:
        1. Pathophysiology and causes
        2. Typical characteristics and appearance
        3. Standard treatment protocols
        4. Expected healing timeline
        5. Potential complications
        6. How the current tissue composition aligns with this wound type
        
        Make it comprehensive for clinical reference.
        
        CRITICAL: Do not use ANY markdown formatting including asterisks (*), double asterisks (**), 
        underscores (_), hash symbols (#), or any other markdown syntax. 
        Use only plain text with clear formatting and proper paragraphs.
        """
        
        chat = gemini_model.start_chat()
        response = chat.send_message(prompt)
        
        # Clean up any markdown formatting aggressively
        cleaned_response = clean_markdown_formatting(response.text)
        
        return cleaned_response
        
    except Exception as e:
        return f"Classification information unavailable: {str(e)}"

def generate_clinical_recommendations(tissue_data, wound_type, health_score):
    """Generate clinical recommendations using Gemini AI"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return ["Clinical recommendations unavailable - AI service not available"]
        
        prompt = f"""
        As a wound care specialist, provide specific clinical recommendations for this wound:

        Analysis Results:
        - Wound Type: {wound_type}
        - Health Score: {health_score}/100
        - Tissue Composition: {format_tissue_data_for_prompt(tissue_data)}
        
        Provide actionable clinical recommendations for:
        1. Immediate wound care interventions
        2. Dressing selection and frequency
        3. Infection prevention strategies
        4. Patient education points
        5. Follow-up monitoring schedule
        6. When to escalate care
        
        Format as numbered recommendations that healthcare providers can implement.
        
        CRITICAL: Do not use ANY markdown formatting including asterisks (*), double asterisks (**), 
        underscores (_), hash symbols (#), or any other markdown syntax. 
        Use only plain text with clear numbering and proper sentences.
        """
        
        chat = gemini_model.start_chat()
        response = chat.send_message(prompt)
        
        # Clean markdown formatting aggressively
        cleaned_text = clean_markdown_formatting(response.text)
        
        # Parse the response into a list of recommendations
        recommendations = []
        lines = cleaned_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Additional cleaning for any remaining markdown
                line = clean_markdown_formatting(line)
                recommendations.append(line)
        
        return recommendations if recommendations else [clean_markdown_formatting(response.text)]
        
    except Exception as e:
        return [f"Clinical recommendations unavailable: {str(e)}"]

def format_tissue_data_for_prompt(tissue_data):
    """Format tissue data for AI prompts"""
    formatted = []
    for tissue, info in tissue_data.items():
        if tissue == "background":
            continue
        if info['percentage'] > 0:
            formatted.append(f"- {tissue.title()}: {info['percentage']:.1f}% ({info['area_px']:,} pixels)")
    return '\n'.join(formatted)

def generate_ai_health_score(tissue_data, wound_type):
    """Generate AI health score independently using Gemini AI"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return 50, "AI service unavailable - using neutral score"
        
        prompt = f"""
        As a wound healing expert, analyze this wound and provide a comprehensive health score assessment:

        Wound Analysis Data:
        - Wound Type: {wound_type}
        - Tissue Composition: {format_tissue_data_for_prompt(tissue_data)}
        
        Based on your clinical expertise, evaluate:
        1. Tissue composition quality and healing indicators
        2. Wound type-specific healing expectations
        3. Overall healing trajectory and prognosis
        4. Risk factors and complications
        5. Tissue balance and regeneration potential
        
        Provide a health score from 0-100 where:
        - 90-100: Excellent healing, optimal tissue composition
        - 80-89: Good healing progress, favorable indicators
        - 70-79: Moderate healing, some positive signs
        - 60-69: Fair healing, mixed indicators
        - 50-59: Poor healing, concerning factors
        - 40-49: Very poor healing, significant issues
        - 0-39: Critical condition, immediate intervention needed
        
        Format your response as:
        SCORE: [number 0-100]
        JUSTIFICATION: [detailed clinical reasoning for the score]
        
        Base your assessment purely on clinical wound healing principles and tissue analysis.
        
        CRITICAL: Do not use ANY markdown formatting including asterisks (*), double asterisks (**), 
        underscores (_), hash symbols (#), or any other markdown syntax.
        """
        
        chat = gemini_model.start_chat()
        response = chat.send_message(prompt)
        
        # Clean markdown formatting
        cleaned_response = clean_markdown_formatting(response.text)
        
        # Parse the response to extract score and justification
        lines = cleaned_response.split('\n')
        ai_score = 50  # Default neutral score
        justification = "AI analysis completed"
        
        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    score_text = line.split(':')[1].strip()
                    # Extract just the number if there are additional words
                    import re
                    score_match = re.search(r'\d+', score_text)
                    if score_match:
                        ai_score = int(score_match.group())
                        # Ensure score is within valid range
                        ai_score = max(0, min(100, ai_score))
                except:
                    pass
            elif line.startswith('JUSTIFICATION:'):
                justification = line.split(':', 1)[1].strip()
        
        return ai_score, justification
        
    except Exception as e:
        return 50, f"AI assessment unavailable: {str(e)}"

def generate_professional_report(tissue_data, wound_type, confidence, health_score, recommendations):
    """Generate a comprehensive professional wound report using Gemini AI"""
    try:
        gemini_model = initialize_gemini()
        if not gemini_model:
            return "Professional report unavailable - AI service not available"
        
        prompt = f"""
        Generate a comprehensive professional wound assessment report:

        WOUND ANALYSIS DATA:
        - Wound Classification: {wound_type} (Confidence: {confidence:.1%})
        - Health Score: {health_score}/100
        - Tissue Composition Analysis: {format_tissue_data_for_prompt(tissue_data)}
        
        Create a detailed clinical report including:
        
        1. EXECUTIVE SUMMARY
        2. WOUND ASSESSMENT FINDINGS
        3. TISSUE COMPOSITION ANALYSIS
        4. CLASSIFICATION RATIONALE
        5. HEALING ASSESSMENT
        6. RISK STRATIFICATION
        7. TREATMENT RECOMMENDATIONS
        8. MONITORING PROTOCOL
        9. PATIENT EDUCATION POINTS
        10. FOLLOW-UP SCHEDULE
        
        Format as a professional medical report suitable for clinical documentation.
        Include specific measurements, percentages, and clinical terminology.
        Emphasize evidence-based recommendations and standard care protocols.
        
        CRITICAL: Do not use ANY markdown formatting including asterisks (*), double asterisks (**), 
        underscores (_), hash symbols (#), or any other markdown syntax. 
        Use only plain text with clear section headers and proper paragraph formatting.
        """
        
        chat = gemini_model.start_chat()
        response = chat.send_message(prompt)
        
        # Clean up any markdown formatting aggressively
        cleaned_response = clean_markdown_formatting(response.text)
        
        return cleaned_response
        
    except Exception as e:
        return f"Professional report generation failed: {str(e)}"

def clean_markdown_formatting(text):
    """Clean markdown formatting from text"""
    import re
    cleaned = text
    # Remove markdown patterns
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove **bold**
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)      # Remove *italic*
    cleaned = re.sub(r'_(.*?)_', r'\1', cleaned)        # Remove _underline_
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)         # Remove # headers
    cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)        # Remove `code`
    return cleaned

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

# Theme toggle button
col1, col2, col3 = st.columns([1, 8, 1])
with col3:
    if st.button("🌓", help="Toggle theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.experimental_rerun()

# Get theme colors AFTER session state is initialized
COL = get_theme_colors()

# Enhanced CSS with theme support (keeping the same CSS as before)
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
    color: #ffffff;
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
  
  /* Remove black boxes from tabs */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    background: transparent !important;
  }}
  
  .stTabs [data-baseweb="tab"] {{
    height: 50px;
    white-space: pre-wrap;
    background-color: transparent !important;
    border: 2px solid {COL['accent']} !important;
    border-radius: 15px !important;
    color: {COL['text_primary']} !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    margin: 0 5px !important;
    transition: all 0.3s ease !important;
  }}
  
  .stTabs [data-baseweb="tab"]:hover {{
    background-color: {COL['accent']} !important;
    color: white !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
  }}
  
  .stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {COL['gradient_start']}, {COL['gradient_end']}) !important;
    color: white !important;
    border-color: {COL['highlight']} !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
  }}
  
  .stTabs [data-baseweb="tab-panel"] {{
    background: transparent !important;
    padding: 0 !important;
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
  
  /* Professional Report Styling */
  .report-container {{
    background: {"linear-gradient(145deg, " + COL['dark'] + " 0%, #2a4a37 100%)" if st.session_state.dark_mode else "linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%)"};
    border-radius: 15px;
    padding: 30px;
    margin: 25px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.1);
    border: 1px solid {COL['border_color']};
    color: {COL['text_primary']};
    font-family: 'Georgia', serif;
    line-height: 1.6;
  }}
  
  .report-container h1, .report-container h2, .report-container h3 {{
    color: {COL['highlight']};
    margin-top: 25px;
    margin-bottom: 15px;
  }}
  
  .report-container h1 {{
    font-size: 1.8rem;
    border-bottom: 2px solid {COL['accent']};
    padding-bottom: 10px;
  }}
  
  .report-container h2 {{
    font-size: 1.4rem;
  }}
  
  .report-container h3 {{
    font-size: 1.2rem;
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

# ──── Tissue Analysis Model Functions ────────────────────────────────────────────
@st.cache_resource
def load_tissue_model():
    """Load tissue analysis model with caching to prevent reloading"""
    try:
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
        st.error(f"Error loading tissue analysis model: {str(e)}")
        # Create a dummy model for demonstration
        st.warning("Using fallback mode for tissue analysis")
        
        class DummyModel:
            def __init__(self):
                pass
                
            def __call__(self, x):
                # Create a dummy tensor with the right shape
                batch_size = x.shape[0]
                h, w = IMG_SIZE, IMG_SIZE
                dummy_output = torch.zeros((batch_size, N_CLASSES, h, w))
                # Make class 0 (background) and class 1 (fibrin) more likely in different areas
                dummy_output[:, 0, :h//2, :] = 5.0  # background in top half
                dummy_output[:, 1, h//2:, :] = 5.0  # fibrin in bottom half
                dummy_output[:, 2, h//3:2*h//3, w//3:2*w//3] = 7.0  # granulation in middle
                return dummy_output
            
            def eval(self):
                return self
                
        return DummyModel()

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
    wound_pixels = 0
    tissue_pixel_counts = {}
    for idx, name in enumerate(class_names):
        # Skip unused classes
        if name not in DISPLAY_CLASSES:
            continue
        if name == "background":
            continue
        class_pixels = np.sum(mask == idx)
        if class_pixels > 0:
            tissue_pixel_counts[name] = class_pixels
            wound_pixels += class_pixels
    data = {}
    for name, pixels in tissue_pixel_counts.items():
        data[name] = { 
            'percentage': (pixels / wound_pixels) * 100 if wound_pixels > 0 else 0,
            'area_px': int(pixels)
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

def calculate_open_defect_area(tissue_data):
    """Calculate open defect area as sum of fibrin and granulation pixels"""
    fibrin_area = tissue_data.get("fibrin", {}).get('area_px', 0)
    granulation_area = tissue_data.get("granulation", {}).get('area_px', 0)
    return fibrin_area + granulation_area

# ──── Wound Classification Model Functions ──────────────────────────────────────
@st.cache_resource
def load_classification_model():
    """Load wound classification model with caching to prevent reloading"""
    try:
        # First attempt - standard loading
        return load_learner(str(CLASSIFICATION_MODEL_PATH))
    except Exception as e:
        st.error(f"Error loading classification model: {str(e)}")
        
        # Create a dummy classification model
        st.warning("Using fallback mode for wound classification")
        
        class DummyClassifier:
            def __init__(self):
                self.classes = ["pressure_injury", "venous_ulcer", "diabetic_foot_ulcer", 
                                "arterial_ulcer", "surgical_wound", "burn"]
            
            def predict(self, img):
                # Always predict pressure injury with 80% confidence
                pred_class = "pressure_injury"
                pred_idx = 0
                # Create a dummy output tensor with confidence scores
                outputs = torch.tensor([0.8, 0.05, 0.05, 0.05, 0.03, 0.02])
                return pred_class, pred_idx, outputs
        
        return DummyClassifier()

def get_models():
    """Get cached models without session state"""
    tissue_model = load_tissue_model()
    classification_model = load_classification_model()
    return tissue_model, classification_model

# ──── Page Layout ────────────────────────────────────────────────
st.markdown('<div class="content-wrapper">', unsafe_allow_html=True)

# ──── Header ───────────────────────────────────────────────────
if LOGO_PATH.exists():
    try:
        logo_data = open(str(LOGO_PATH), 'rb').read()
        st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{base64.b64encode(logo_data).decode()}" class="logo">
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading logo: {str(e)}")

st.markdown("""
<div class="header">
  <h1>🩹 Advanced Wound Analysis</h1>
  <p>Professional AI-Powered Wound Assessment & Tissue Composition Analysis with Gemini AI</p>
</div>
""", unsafe_allow_html=True)

# ──── Instructions ─────────────────────────────────────────────
st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
st.markdown("""
<div class="instructions">
  <strong>🔬 Advanced Wound Analysis System:</strong><br>
  <ol>
    <li><b>Upload</b> a clear wound image (PNG/JPG/JPEG)</li>
    <li><b>Analyze</b> to get comprehensive tissue composition analysis and wound classification</li>
    <li><b>View</b> detailed results with AI-enhanced assessments and professional recommendations</li>
    <li><b>Generate</b> professional reports for clinical documentation</li>
    <li><b>Monitor</b> wound progress over time with professional-grade assessment</li>
  </ol>
  <p><strong>Note:</strong> The "background" classification refers to non-wound areas in the image and is not part of the actual wound tissue analysis.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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
    try:
        # Add missing import for pandas at the top if not already imported
        try:
            import pandas as pd
            timestamp_str = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        except ImportError:
            # Fallback if pandas not available
            from datetime import datetime
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Read and resize image if needed to reduce memory usage
        pil_img = Image.open(uploaded).convert("RGB")
        orig_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Resize if too large
        orig_bgr = resize_image_if_needed(orig_bgr, MAX_IMAGE_SIZE)
        pil_img = Image.fromarray(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB))
        
        # Clear memory
        clear_memory()

        # Display uploaded image
        st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(pil_img, caption="Uploaded Wound Image")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Analysis button
        st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
        if st.button("🚀 Analyze Wound", help="Click to run comprehensive AI analysis"):
            # Get cached models
            with st.spinner("Initializing AI models..."):
                tissue_model, classification_model = get_models()
                gemini_model = initialize_gemini()
            
            # ──── Complete Analysis ────────────────────────────────────────────────
            with st.spinner("Running comprehensive wound analysis..."):
                progress = st.progress(0)

                # Step 1: Wound classification
                for i in range(30):
                    progress.progress(i+1)
                
                pred_class, pred_idx, outputs = classification_model.predict(pil_img)
                confidence = outputs[pred_idx].item()

                # Step 2: Tissue analysis
                for i in range(30, 60):
                    progress.progress(i+1)

                with torch.no_grad():
                    tensor_img = preprocess_tissue(pil_img)
                    tissue_pred = tissue_model(tensor_img)
                    tissue_mask_bgr, tissue_mask_indices = postprocess_tissue(tissue_pred)
                    tissue_data = calculate_tissue_percentages_and_areas(tissue_mask_indices, CLASS_NAMES)
                    
                    # Clear intermediate tensors
                    del tensor_img, tissue_pred

                # Step 3: Basic calculations
                for i in range(60, 70):
                    progress.progress(i+1)

                basic_health_score = calculate_health_score(tissue_data)
                dominant_tissue, dominant_percent = get_dominant_tissue(tissue_data)
                open_defect_area = calculate_open_defect_area(tissue_data)

                # Step 4: AI Enhancement with Gemini
                for i in range(70, 100):
                    progress.progress(i+1)

                # Generate AI-enhanced assessments
                if gemini_model:
                    with st.spinner("Generating AI-enhanced assessments..."):
                        # AI-generated health score (independent)
                        ai_health_score, health_justification = generate_ai_health_score(
                            tissue_data, pred_class
                        )
                        
                        # AI recommendations
                        ai_recommendations = generate_clinical_recommendations(
                            tissue_data, pred_class, ai_health_score
                        )
                        
                        # Health assessment
                        health_assessment = generate_health_assessment(
                            tissue_data, pred_class, confidence
                        )
                        
                        # Classification information
                        classification_info = generate_wound_classification_info(
                            pred_class, confidence, tissue_data
                        )
                else:
                    ai_health_score = basic_health_score
                    health_justification = "AI service unavailable - using basic calculation"
                    ai_recommendations = generate_recommendations(tissue_data)
                    health_assessment = "AI assessment unavailable"
                    classification_info = "AI classification info unavailable"

                progress.empty()
                clear_memory()

                st.success("✅ Complete AI-enhanced analysis finished!")
                st.markdown('<div class="results-header">Advanced Wound Analysis Results</div>', unsafe_allow_html=True)

                # ──── Image Results Display ────────────────────────────────────────────────
                st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                # Prepare images for display
                tissue_display = cv2.cvtColor(tissue_mask_bgr, cv2.COLOR_BGR2RGB)

                with col1:
                    st.markdown('<div class="img-container">', unsafe_allow_html=True)
                    st.image(tissue_display, caption="🧬 Tissue Composition Analysis")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="img-container">', unsafe_allow_html=True)
                    # Combined overlay
                    alpha = 0.5
                    orig_bgr_resized = cv2.resize(orig_bgr, (IMG_SIZE, IMG_SIZE))
                    tissue_overlay = cv2.addWeighted(orig_bgr_resized, 1 - alpha, tissue_mask_bgr, alpha, 0)
                    tissue_overlay_rgb = cv2.cvtColor(tissue_overlay, cv2.COLOR_BGR2RGB)
                    st.image(tissue_overlay_rgb, caption="🔗 Combined Analysis Overlay")
                    st.markdown('</div>', unsafe_allow_html=True)
    
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Clean up images
                del tissue_display, orig_bgr_resized, tissue_overlay, tissue_overlay_rgb
                clear_memory()

                # ──── Key Metrics Dashboard ────────────────────────────────────────────────
                st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
                col1, col2, col3, col4, col5 = st.columns(5)
                tissue_types_count = len([t for t in tissue_data.keys() if t != "background" and tissue_data[t]['percentage'] > 0])
                dominant_tissue, dominant_percent = get_dominant_tissue(tissue_data)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{ai_health_score:.0f}</div>
                        <div class="metric-label">AI Health Score</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{open_defect_area:,}</div>
                        <div class="metric-label">Open Defect Area (px)</div>
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
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{tissue_types_count}</div>
                        <div class="metric-label">Tissue Types</div>
                    </div>    
                    """, unsafe_allow_html=True)

                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{pred_class}</div>
                        <div class="metric-label">Wound Type</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # ──── Professional Report Button ────────────────────────────────────────────────
                st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
                if st.button("📋 Generate Professional Report", help="Generate comprehensive clinical report"):
                    with st.spinner("Generating professional wound assessment report..."):
                        professional_report = generate_professional_report(
                            tissue_data, pred_class, confidence, ai_health_score, ai_recommendations
                        )
                        
                        # Display the report
                        st.markdown('<div class="report-container">', unsafe_allow_html=True)
                        st.markdown("**📋 Professional Wound Assessment Report**")
                        st.write(professional_report)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Download button for the report
                        st.download_button(
                            label="📥 Download Report",
                            data=professional_report,
                            file_name=f"wound_assessment_report_{timestamp_str}.txt",
                            mime="text/plain"
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)

                # ──── Detailed Analysis Tabs with AI Enhancement ────────────────────────────────────────────────
                tab1, tab2, tab3, tab4 = st.tabs(["🧬 Tissue Composition", "📊 AI Health Assessment", "🏥 AI Wound Classification", "💡 AI Clinical Recommendations"])

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
                    st.markdown('<div class="tab-title">AI-Enhanced Health Assessment</div>', unsafe_allow_html=True)

                    # Enhanced health score interpretation
                    if ai_health_score >= 80:
                        health_status = "Excellent"
                        health_color = COL['success']
                        health_icon = "🌟"
                    elif ai_health_score >= 60:
                        health_status = "Good"
                        health_color = COL['success']
                        health_icon = "✅"
                    elif ai_health_score >= 40:
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
                        <div style="font-size: 2.5rem; font-weight: 800; color: {health_color};">{ai_health_score:.0f}/100</div>
                        <div style="font-size: 1.5rem; margin-top: 10px;">AI Health Assessment: {health_status}</div>
                        <div style="font-size: 1.1rem; margin-top: 15px; opacity: 0.9;">{health_justification}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # AI-generated detailed assessment
                    st.markdown("**Detailed AI Health Assessment:**")
                    st.markdown(f"""
                    <div style="background: {COL['card_bg']}; padding: 20px; border-radius: 10px; 
                        margin: 20px 0; border: 1px solid {COL['border_color']}; color: {COL['text_primary']};">
                        {health_assessment}
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with tab3:
                    st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
                    st.markdown('<div class="tab-title">AI Wound Classification Analysis</div>', unsafe_allow_html=True)
                    
                    # Classification confidence display
                    st.markdown(f"""
                    <div style="text-align: center; padding: 20px; background: {COL['card_bg']}; 
                        border-radius: 10px; margin: 20px 0; border: 1px solid {COL['border_color']};">
                        <div style="font-size: 1.8rem; font-weight: 700; color: {COL['highlight']};">
                            {pred_class.replace('_', ' ').title()}
                        </div>
                        <div style="font-size: 1.2rem; margin-top: 10px; color: {COL['text_primary']};">
                            AI Confidence: {confidence:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # AI-generated classification information
                    st.markdown("**AI Classification Analysis:**")
                    st.markdown(f"""
                    <div style="background: {COL['card_bg']}; padding: 20px; border-radius: 10px; 
                        margin: 20px 0; border: 1px solid {COL['border_color']}; color: {COL['text_primary']};">
                        {classification_info}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                with tab4:
                    st.markdown('<div class="analysis-tab">', unsafe_allow_html=True)
                    st.markdown('<div class="tab-title">AI Clinical Recommendations</div>', unsafe_allow_html=True)
                    st.markdown("**AI Clinical Recommendations:**")
                    recommendations_text = "\n\n".join([f"{i}. {rec}" for i, rec in enumerate(ai_recommendations, 1)])
                    st.markdown(f"""
                    <div style="background: {COL['card_bg']}; padding: 20px; border-radius: 10px;
                        margin: 20px 0; border: 1px solid {COL['border_color']}; color: {COL['text_primary']};">
                        {recommendations_text}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    


        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.write("Exception details:")
        st.exception(e)
        clear_memory()

# ──── Footer ────────────────────────────────────────────────────
st.markdown('</div>', unsafe_allow_html=True)  # Close content-wrapper

st.markdown("""
<div class="footer">
    <strong>Advanced Wound Analysis System with Gemini AI</strong><br>
    Powered by deep learning models and Google Gemini AI for comprehensive wound assessment and monitoring.<br>
    <em>For research and educational purposes. Always consult healthcare professionals for medical decisions.</em>
</div>
""", unsafe_allow_html=True)
