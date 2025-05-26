import streamlit as st
import torch
import gdown
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os

st.set_page_config(page_title="Wound Segmentation Demo", layout="centered")
st.title("🩹 Wound Segmentation Model Demo")
st.write("Upload an image of a wound to segment with the pretrained model.")

# ---- MODEL CONFIG ----
MODEL_DRIVE_ID = "1q0xk9wll0eyF3-CKEc5s6MfG0gE_jde1"
MODEL_FILENAME = "best_model_streamlit.pth"
N_CLASSES = 9
ENCODER = "mit_b3"
INPUT_SIZE = 256  # Change to 512 if your model uses 512x512
CLASS_NAMES = [
    "background", "granulation", "callus", "fibrin", "necrotic", "eschar", "neodermis", "tendon", "dressing"
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

# ---- Download model weights if needed ----
if not os.path.exists(MODEL_FILENAME):
    st.info("Downloading model weights...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}", MODEL_FILENAME, quiet=False)

# ---- Model loader ----
@st.cache_resource(show_spinner=False)
def load_model():
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=N_CLASSES,
        decoder_attention_type='scse',
        activation=None,
    )
    state_dict = torch.load(MODEL_FILENAME, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess(image_pil):
    # Resize and normalize
    image = np.array(image_pil.resize((INPUT_SIZE, INPUT_SIZE))) / 255.0
    image = image.transpose(2, 0, 1)  # HWC -> CHW
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return image

def postprocess(mask):
    # mask shape: (1, C, H, W) or (C, H, W)
    if mask.ndim == 4:
        mask = mask.squeeze(0)
    mask = mask.argmax(0).cpu().numpy()
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        color_mask[mask == idx] = color
    return color_mask, mask

def calculate_tissue_percentages(mask, class_names):
    total_pixels = mask.size
    percentages = {}
    for idx, name in enumerate(class_names):
        class_pixels = np.sum(mask == idx)
        if class_pixels > 0:
            percentages[name] = (class_pixels / total_pixels) * 100
    return percentages

# ---- Streamlit App ----

uploaded_file = st.file_uploader("Upload a wound image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="Original Image", use_column_width=True)

    with st.spinner("Running segmentation..."):
        model = load_model()
        input_tensor = preprocess(image_pil)
        with torch.no_grad():
            output = model(input_tensor)
        mask_img, class_mask = postprocess(output)

    st.image(mask_img, caption="Predicted Segmentation", use_column_width=True)

    # ---- Wound Tissue Composition ----
    tissue_percent = calculate_tissue_percentages(class_mask, CLASS_NAMES)
    st.markdown("### Wound Tissue Composition:")
    comp_str = ""
    for name, percent in sorted(tissue_percent.items(), key=lambda x: -x[1]):
        comp_str += f"**{name}**: {percent:.2f}%  \n"
    st.markdown(comp_str)

    # Optional: Overlay mask on original
    if st.checkbox("Show mask overlay"):
        orig = np.array(image_pil.resize((INPUT_SIZE, INPUT_SIZE)))
        overlay = (0.6 * orig + 0.4 * mask_img).astype(np.uint8)
        st.image(overlay, caption="Overlay", use_column_width=True)
