import streamlit as st
import torch
import gdown
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os
import tensorflow as tf
import cv2

# --------- CONFIG ---------
# PyTorch tissue model (multiclass)
PT_MODEL_DRIVE_ID = "1q0xk9wll0eyF3-CKEc5s6MfG0gE_jde1"
PT_MODEL_FILENAME = "best_model_streamlit.pth"
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

# Keras wound segmentation model (binary)
K_MODEL_DRIVE_ID = "1_PToBgQjEKAQAZ9ZX10sRpdgxQ18C-18"
K_MODEL_FILENAME = "unet_wound_segmentation_best.h5"

st.set_page_config(page_title="Wound Segmentation Demo", layout="centered")
st.title("🩹 Wound Segmentation Model Demo")
st.write("Upload an image of a wound to analyze with the AI models.")

# ---- Download models if needed ----
if not os.path.exists(PT_MODEL_FILENAME):
    st.info("Downloading PyTorch tissue model...")
    gdown.download(f"https://drive.google.com/uc?id={PT_MODEL_DRIVE_ID}", PT_MODEL_FILENAME, quiet=False)

if not os.path.exists(K_MODEL_FILENAME):
    st.info("Downloading Keras wound segmentation model...")
    gdown.download(f"https://drive.google.com/uc?id={K_MODEL_DRIVE_ID}", K_MODEL_FILENAME, quiet=False)

# ---- Load Keras segmentation model ----
@st.cache_resource(show_spinner=False)
def load_keras_model():
    return tf.keras.models.load_model(K_MODEL_FILENAME, compile=False)

# ---- Load PyTorch tissue model ----
@st.cache_resource(show_spinner=False)
def load_pytorch_model():
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=N_CLASSES,
        decoder_attention_type='scse',
        activation=None,
    )
    state_dict = torch.load(PT_MODEL_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---- Preprocessing and Postprocessing ----
def preprocess_for_keras(img_pil):
    img = np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE)))
    img = img.astype("float32") / 255.
    img = np.expand_dims(img, 0)  # (1, H, W, C)
    return img

def get_wound_mask(keras_model, img_pil):
    img = preprocess_for_keras(img_pil)
    pred = keras_model.predict(img, verbose=0)[0, ..., 0]
    wound_mask = (pred > 0.5).astype(np.uint8)
    return wound_mask  # (H, W), binary mask

def preprocess_for_pytorch(img_pil):
    img = np.array(img_pil.resize((INPUT_SIZE, INPUT_SIZE))) / 255.0
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def postprocess(mask):
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    class_map = mask.argmax(0).cpu().numpy()  # shape: (H, W)
    color_mask = np.zeros((class_map.shape[0], class_map.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        color_mask[class_map == idx] = color
    return color_mask, class_map

def compute_class_percentages(class_map, wound_mask):
    percentages = {}
    # Only count within wound region
    total_wound_pixels = np.sum(wound_mask)
    if total_wound_pixels == 0:
        return {name: 0.0 for name in CLASS_NAMES}
    for idx, name in enumerate(CLASS_NAMES):
        tissue_pixels = np.sum((class_map == idx) & (wound_mask == 1))
        percent = 100 * tissue_pixels / total_wound_pixels
        if percent > 0.1:  # only show if >0.1% (adjust if needed)
            percentages[name] = percent
    return percentages

# ---- Streamlit App ----

uploaded_file = st.file_uploader("Upload a wound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Original Image", use_column_width=True)

    with st.spinner("Running wound segmentation..."):
        keras_model = load_keras_model()
        wound_mask = get_wound_mask(keras_model, image_pil)  # binary mask

        # Visualize wound mask
        wound_mask_rgb = np.stack([wound_mask*255]*3, axis=-1)
        st.image(wound_mask_rgb, caption="Wound Region (Binary Mask)", use_column_width=True)

    # Apply mask to the image for tissue classification
    img_arr = np.array(image_pil.resize((INPUT_SIZE, INPUT_SIZE)))
    img_masked = img_arr.copy()
    img_masked[wound_mask == 0] = 0  # Zero out non-wound

    with st.spinner("Running tissue classification..."):
        pytorch_model = load_pytorch_model()
        input_tensor = preprocess_for_pytorch(Image.fromarray(img_masked))
        with torch.no_grad():
            output = pytorch_model(input_tensor)
        mask_img, class_map = postprocess(output)

    st.image(mask_img, caption="Predicted Tissue Segmentation", use_column_width=True)

    # ----- Show tissue composition only within wound area -----
    percentages = compute_class_percentages(class_map, wound_mask)
    st.markdown("### Wound Tissue Composition:")
    for name, percent in sorted(percentages.items(), key=lambda x: -x[1]):
        st.write(f"**{name}**: {percent:.2f}%")

    # Optional: Overlay mask on original
    if st.checkbox("Show mask overlay"):
        orig = np.array(image_pil.resize((INPUT_SIZE, INPUT_SIZE)))
        overlay = (0.6 * orig + 0.4 * mask_img).astype(np.uint8)
        st.image(overlay, caption="Overlay", use_column_width=True)
