"""
app.py  —  Oral Lesion Segmentation GUI
Run: streamlit run app.py
"""

import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from scipy import ndimage
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="OralSeg — Oral Lesion Detector",
    page_icon="🦷",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
body, [class*="css"] { font-family: 'Inter', sans-serif; }

.result-box {
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    font-size: 18px;
    font-weight: 600;
    text-align: center;
}
.result-normal  { background:#dcfce7; color:#166534; border:2px solid #86efac; }
.result-disease { background:#fee2e2; color:#991b1b; border:2px solid #fca5a5; }
.result-warning { background:#fef9c3; color:#854d0e; border:2px solid #fde047; }

.metric-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    margin: 4px;
}
.metric-label { font-size:11px; color:#64748b; text-transform:uppercase;
                letter-spacing:.08em; margin-bottom:4px; }
.metric-value { font-size:22px; font-weight:700; color:#1e40af; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
IMG_SIZE    = 512
MODEL_REPO  = "zareejassy/oral-lesion-project"
MODEL_FILE  = "best_model.pth"
_MEAN       = np.array([0.485, 0.456, 0.406])
_STD        = np.array([0.229, 0.224, 0.225])

# ── Calibrated from CELL 16 of training — update after retraining ──
# Replace this value with the best_thresh printed in CELL 16
NO_DISEASE_THRESHOLD_PCT = 0.5

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    conf_thresh   = st.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05,
                              help="Minimum probability to classify a pixel as lesion")
    overlay_alpha = st.slider("Overlay opacity",      0.20, 0.90, 0.50, 0.05)
    st.markdown("---")
    st.markdown("**Post-processing**")
    remove_small  = st.checkbox("Remove small fragments", value=True)
    min_blob_pct  = st.slider("Min lesion size (% image)", 0.1, 5.0, 0.5, 0.1,
                              disabled=not remove_small)
    smooth_mask   = st.checkbox("Smooth mask edges", value=True)
    st.markdown("---")
    st.caption(f"No-Disease threshold: {NO_DISEASE_THRESHOLD_PCT}%")
    st.caption("Model: SegFormer-B2 · 27.35M params")
    st.caption("Classes: background / lesion")


# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    from huggingface_hub import hf_hub_download

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": IMG_SIZE, "width": IMG_SIZE},
        do_normalize=True,
        do_reduce_labels=False,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2",
        num_labels=2,
        id2label={0: "background", 1: "lesion"},
        label2id={"background": 0, "lesion": 1},
        ignore_mismatched_sizes=True,
    )

    ckpt = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, processor, device


# ── Inference ─────────────────────────────────────────────────
def run_inference(image_pil, conf, model, processor, device):
    """Returns (binary_mask, prob_map) both at original resolution."""
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    orig_w, orig_h = image_pil.size
    img_np = np.array(image_pil)

    inputs = processor(images=img_np, return_tensors="pt")
    pv     = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(pixel_values=pv).logits
        up     = F.interpolate(logits, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
        probs  = up.softmax(dim=1)[0, 1].cpu().numpy()   # P(lesion)

    mask = (probs >= conf).astype(np.uint8)
    return mask, probs


def postprocess(mask, img_shape, remove_small, min_pct, smooth):
    """Clean binary mask."""
    if smooth:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    if remove_small:
        H, W     = img_shape[:2]
        min_area = int((min_pct / 100.0) * H * W)
        lbl, n   = ndimage.label(mask)
        for c in range(1, n + 1):
            if (lbl == c).sum() < min_area:
                mask[lbl == c] = 0
    return mask


def make_overlay(image_pil, mask, alpha):
    """Red overlay + green contour on lesion."""
    base    = np.array(image_pil.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[mask == 1, 0] = 255
    overlay[mask == 1, 1] = overlay[mask == 1, 1] * 0.15
    overlay[mask == 1, 2] = overlay[mask == 1, 2] * 0.15
    blended  = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 220, 60), 2)
    return Image.fromarray(blended)


def get_diagnosis(mask, orig_w, orig_h, threshold_pct):
    """Return diagnosis dict."""
    total_px  = orig_h * orig_w
    lesion_px = int(mask.sum())
    lesion_pct = lesion_px / total_px * 100
    lbl, n_regions = ndimage.label(mask)

    is_normal = lesion_pct < threshold_pct

    if is_normal:
        status  = "No Disease Detected"
        detail  = "The oral cavity appears healthy. No significant lesion regions were identified."
        css     = "result-normal"
        emoji   = "✅"
    elif lesion_pct < 5.0:
        status  = "Minor Lesion Detected"
        detail  = (f"A small lesion region was identified covering {lesion_pct:.1f}% of the image. "
                   "Clinical examination is recommended.")
        css     = "result-warning"
        emoji   = "⚠️"
    else:
        status  = "Significant Lesion Detected"
        detail  = (f"Lesion regions detected covering {lesion_pct:.1f}% of the image "
                   f"({n_regions} region{'s' if n_regions > 1 else ''}). "
                   "Please consult a dental specialist.")
        css     = "result-disease"
        emoji   = "🔴"

    return {
        "status":      status,
        "detail":      detail,
        "css":         css,
        "emoji":       emoji,
        "lesion_pct":  lesion_pct,
        "lesion_px":   lesion_px,
        "n_regions":   n_regions,
        "is_normal":   is_normal,
    }


# ── Main UI ───────────────────────────────────────────────────
st.markdown("# 🦷 Oral Lesion Segmentation")
st.markdown(
    "Upload an intraoral photograph. "
    "The AI will detect and segment oral lesions, "
    "or confirm no disease if the mouth appears healthy."
)

# Model load
try:
    model, processor, device = load_model()
    st.success("Model loaded ✅", icon="✅")
    model_ready = True
except Exception as e:
    st.error(f"Model load failed: {e}")
    model_ready = False

st.markdown("---")

uploaded = st.file_uploader(
    "Upload intraoral image (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded and model_ready:
    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"Cannot read image: {e}")
        st.stop()

    orig_w, orig_h = image.size

    with st.spinner("Running segmentation…"):
        raw_mask, prob_map = run_inference(image, conf_thresh, model, processor, device)
        clean_mask         = postprocess(raw_mask.copy(), (orig_h, orig_w),
                                         remove_small, min_blob_pct, smooth_mask)
        diagnosis          = get_diagnosis(clean_mask, orig_w, orig_h,
                                           NO_DISEASE_THRESHOLD_PCT)
        overlay_img        = make_overlay(image, clean_mask, overlay_alpha)

    # ── Diagnosis banner ────────────────────────────────────
    st.markdown(
        f'<div class="result-box {diagnosis["css"]}">'
        f'{diagnosis["emoji"]} &nbsp; {diagnosis["status"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"**{diagnosis['detail']}**")
    st.markdown("")

    # ── Images ──────────────────────────────────────────────
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("**Segmentation Overlay**")
        if diagnosis["is_normal"]:
            st.image(image, use_container_width=True)
            st.caption("No lesion detected — original image shown.")
        else:
            st.image(overlay_img, use_container_width=True)
            st.caption("Red = lesion area  |  Green = lesion boundary")

    st.markdown("---")

    # ── Metrics ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    pct_color = "#166534" if diagnosis["is_normal"] else (
                "#854d0e" if diagnosis["lesion_pct"] < 5 else "#991b1b")

    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Lesion Coverage</div>
            <div class="metric-value" style="color:{pct_color}">
                {diagnosis['lesion_pct']:.1f}%
            </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Lesion Pixels</div>
            <div class="metric-value">{diagnosis['lesion_px']:,}</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Regions Found</div>
            <div class="metric-value">{diagnosis['n_regions']}</div>
            </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{conf_thresh:.0%}</div>
            </div>""", unsafe_allow_html=True)

    # ── Probability heatmap ──────────────────────────────────
    with st.expander("Show probability heatmap"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        im  = ax.imshow(prob_map, cmap="plasma", vmin=0, vmax=1)
        ax.set_title("P(lesion) per pixel")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.03)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Downloads ────────────────────────────────────────────
    st.markdown("")
    d1, d2 = st.columns(2)
    with d1:
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        st.download_button("⬇️ Download overlay image",
                           buf.getvalue(), "lesion_overlay.png", "image/png")
    with d2:
        buf2 = io.BytesIO()
        Image.fromarray(clean_mask * 255).save(buf2, format="PNG")
        st.download_button("⬇️ Download binary mask",
                           buf2.getvalue(), "lesion_mask.png", "image/png")

elif not model_ready:
    st.warning("Model not loaded. Check your Hugging Face repo.")
else:
    st.info("👆 Upload an intraoral photograph to begin.")

st.markdown("---")
st.caption(
    "SegFormer-B2 · Trained on 882 images (562 disease + 320 normal) · "
    "Based on Zhang et al., Bioengineering 2024"
)
