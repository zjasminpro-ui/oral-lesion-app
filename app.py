"""
app.py  —  OralSeg: Oral Lesion Segmentation & Risk Assessment
Run: streamlit run app.py

Pipeline:
  Stage 1 → SegFormer-B2       : pixel-level lesion segmentation
  Stage 2 → EfficientNet-B0    : 4-class lesion classification
             (Normal / Ulcer / Pre-cancer / Cancer)
  Stage 3 → Risk output        : clinical action recommendation
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

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="OralSeg — Oral Lesion Detector",
    page_icon="🦷",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

.pipeline-wrap {
    display:flex; gap:0; margin:16px 0 24px 0;
    border-radius:12px; overflow:hidden; border:1px solid #e2e8f0;
}
.pipeline-item {
    flex:1; padding:12px 16px; font-size:12px; font-weight:600;
    display:flex; align-items:center; gap:8px; border-right:1px solid #e2e8f0;
}
.pipeline-item:last-child { border-right:none; }
.pi-1 { background:#eff6ff; color:#1d4ed8; }
.pi-2 { background:#fff7ed; color:#c2410c; }
.pi-3 { background:#f5f3ff; color:#7c3aed; }
.pi-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.pi-1 .pi-dot { background:#3b82f6; }
.pi-2 .pi-dot { background:#f97316; }
.pi-3 .pi-dot { background:#8b5cf6; }

.result-banner {
    border-radius:14px; padding:20px 28px; margin:16px 0;
    font-size:19px; font-weight:700; text-align:center; letter-spacing:-0.01em;
}
.rb-normal    { background:#f0fdf4; color:#15803d; border:2px solid #86efac; }
.rb-ulcer     { background:#fff7ed; color:#c2410c; border:2px solid #fdba74; }
.rb-precancer { background:#fffbeb; color:#92400e; border:2px solid #fde68a; }
.rb-cancer    { background:#fef2f2; color:#991b1b; border:2px solid #fca5a5; }

.metric-card {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
    padding:16px; text-align:center;
}
.metric-label {
    font-size:10px; font-weight:600; color:#94a3b8;
    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;
}
.metric-value {
    font-size:24px; font-weight:700; color:#0f172a;
    font-family:'DM Mono', monospace;
}

.status-tags { display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; }
.tag {
    display:inline-block; border-radius:20px;
    padding:4px 12px; font-size:11px; font-weight:700; letter-spacing:0.04em;
}
.tag-green  { background:#dcfce7; color:#15803d; }
.tag-orange { background:#ffedd5; color:#c2410c; }
.tag-yellow { background:#fef9c3; color:#854d0e; }
.tag-red    { background:#fee2e2; color:#991b1b; }
.tag-purple { background:#ede9fe; color:#6d28d9; }
.tag-gray   { background:#f1f5f9; color:#475569; }

.risk-panel {
    border:1px solid #e2e8f0; border-radius:16px;
    padding:24px; margin:12px 0; background:#ffffff;
}
.risk-panel-title {
    font-size:13px; font-weight:700; color:#64748b;
    text-transform:uppercase; letter-spacing:0.08em; margin-bottom:20px;
}
.class-row {
    display:flex; align-items:center; gap:14px;
    margin:12px 0; font-size:13px; font-weight:600;
}
.class-label { min-width:140px; color:#334155; }
.class-bar-outer {
    flex:1; height:14px; background:#f1f5f9; border-radius:7px; overflow:hidden;
}
.class-bar-inner { height:100%; border-radius:7px; }
.class-pct {
    min-width:44px; text-align:right;
    font-family:'DM Mono', monospace; font-size:12px; color:#64748b;
}
.predicted-badge {
    display:inline-block; border-radius:20px;
    padding:2px 10px; font-size:10px; font-weight:700;
    margin-left:6px; text-transform:uppercase; letter-spacing:0.06em;
}
.pb-normal    { background:#dcfce7; color:#166534; }
.pb-ulcer     { background:#ffedd5; color:#c2410c; }
.pb-precancer { background:#fef3c7; color:#92400e; }
.pb-cancer    { background:#fee2e2; color:#991b1b; }

.prediction-box {
    border-radius:12px; padding:16px 22px; margin:0 0 16px 0;
    font-size:17px; font-weight:700; text-align:center; border-left:5px solid;
}
.pred-normal    { background:#f0fdf4; color:#166534; border-color:#22c55e; }
.pred-ulcer     { background:#fff7ed; color:#c2410c; border-color:#f97316; }
.pred-precancer { background:#fffbeb; color:#92400e; border-color:#f59e0b; }
.pred-cancer    { background:#fef2f2; color:#991b1b; border-color:#ef4444; }

.section-header {
    font-size:13px; font-weight:700; color:#64748b;
    text-transform:uppercase; letter-spacing:0.08em;
    margin:24px 0 12px 0; display:flex; align-items:center; gap:8px;
}
.section-line { flex:1; height:1px; background:#f1f5f9; }

.model-status { display:flex; gap:10px; margin:12px 0; }
.ms-pill {
    display:flex; align-items:center; gap:8px;
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:20px; padding:6px 14px; font-size:12px; font-weight:600; color:#374151;
}
.ms-dot-ok   { width:8px; height:8px; border-radius:50%; background:#22c55e; }
.ms-dot-warn { width:8px; height:8px; border-radius:50%; background:#f59e0b; }
.ms-dot-err  { width:8px; height:8px; border-radius:50%; background:#ef4444; }

.panel-note {
    font-size:11px; color:#94a3b8; margin-top:14px;
    padding-top:12px; border-top:1px solid #f1f5f9; line-height:1.6;
}
.disclaimer {
    background:#f8fafc; border-left:3px solid #cbd5e1;
    padding:14px 18px; border-radius:0 10px 10px 0;
    font-size:12px; color:#64748b; margin-top:16px; line-height:1.7;
}
.disclaimer strong { color:#374151; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
IMG_SIZE                 = 512
CROP_SIZE                = 224
MODEL_REPO               = "zareejassy/oral-lesion-project"
SEG_FILE                 = "best_model.pth"
CLF4_FILE                = "classifier4.pth"
CLF2_FILE                = "classifier.pth"
NO_DISEASE_THRESHOLD_PCT = 0.5

CLASS_NAMES  = ["Normal", "Ulcer", "Pre-cancer", "Cancer"]
CLASS_COLORS = ["#16a34a", "#f97316", "#f59e0b", "#ef4444"]
CLASS_BADGES = ["pb-normal", "pb-ulcer", "pb-precancer", "pb-cancer"]
CLASS_PRED   = ["pred-normal", "pred-ulcer", "pred-precancer", "pred-cancer"]
CLASS_ICONS  = ["🟢", "🟠", "🟡", "🔴"]

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_thresh   = st.slider("Confidence threshold", 0.30, 0.90, 0.50, 0.05)
    overlay_alpha = st.slider("Overlay opacity", 0.20, 0.90, 0.50, 0.05)
    st.markdown("---")
    st.markdown("**Post-processing**")
    remove_small = st.checkbox("Remove small fragments", value=True)
    min_blob_pct = st.slider("Min lesion size (%)", 0.1, 5.0, 0.5, 0.1,
                             disabled=not remove_small)
    smooth_mask  = st.checkbox("Smooth mask edges", value=True)
    st.markdown("---")
    show_gradcam = st.checkbox("Attention heatmap", value=True)
    show_crop    = st.checkbox("Cropped lesion patch", value=True)
    st.markdown("---")
    st.caption(f"No-Disease threshold: {NO_DISEASE_THRESHOLD_PCT}%")
    st.caption("Stage 1: SegFormer-B2 · 27.35M params")
    st.caption("Stage 2: EfficientNet-B0 · 5.3M params")


# ── Model loaders ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SegFormer…")
def load_segformer():
    from huggingface_hub import hf_hub_download
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor(
        do_resize=True, size={"height": IMG_SIZE, "width": IMG_SIZE},
        do_normalize=True, do_reduce_labels=False,
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b2", num_labels=2,
        id2label={0: "background", 1: "lesion"},
        label2id={"background": 0, "lesion": 1},
        ignore_mismatched_sizes=True,
    )
    ckpt  = hf_hub_download(repo_id=MODEL_REPO, filename=SEG_FILE)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, processor, device


@st.cache_resource(show_spinner="Loading classifier…")
def load_classifier(device):
    import timm
    import torch.nn as nn
    from huggingface_hub import hf_hub_download

    # Try 4-class model first
    try:
        clf = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0, drop_rate=0.3)
        clf.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(clf.num_features, 4)
        )
        ckpt  = hf_hub_download(repo_id=MODEL_REPO, filename=CLF4_FILE)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        clf.load_state_dict(state, strict=True)
        clf.eval().to(device)
        return clf, "4class"
    except Exception:
        pass

    # Fallback: binary model
    try:
        clf   = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        ckpt  = hf_hub_download(repo_id=MODEL_REPO, filename=CLF2_FILE)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        clf.load_state_dict(state, strict=True)
        clf.eval().to(device)
        return clf, "2class"
    except Exception:
        return None, "none"


# ── Processing functions ──────────────────────────────────────
def run_segmentation(image_pil, conf, model, processor, device):
    if image_pil.mode != "RGB":
        image_pil = image_pil.convert("RGB")
    orig_w, orig_h = image_pil.size
    inputs = processor(images=np.array(image_pil), return_tensors="pt")
    with torch.no_grad():
        logits = model(pixel_values=inputs["pixel_values"].to(device)).logits
        up     = F.interpolate(logits, size=(orig_h, orig_w),
                               mode="bilinear", align_corners=False)
        probs  = up.softmax(dim=1)[0, 1].cpu().numpy()
    return (probs >= conf).astype(np.uint8), probs


def postprocess(mask, img_shape, do_remove, min_pct, do_smooth):
    if do_smooth:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    if do_remove:
        H, W     = img_shape[:2]
        min_area = int((min_pct / 100.0) * H * W)
        lbl, n   = ndimage.label(mask)
        for c in range(1, n + 1):
            if (lbl == c).sum() < min_area:
                mask[lbl == c] = 0
    return mask


def crop_lesion(image_np, mask, pad=20):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    H, W = image_np.shape[:2]
    return image_np[
        max(0, int(ys.min()) - pad):min(H, int(ys.max()) + pad),
        max(0, int(xs.min()) - pad):min(W, int(xs.max()) + pad)
    ]


def get_probs(crop_np, clf, clf_mode, lesion_pct, prob_map, clean_mask, device):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    tf = A.Compose([
        A.Resize(CROP_SIZE, CROP_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    if clf_mode == "4class" and crop_np is not None:
        tensor = tf(image=crop_np)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            return F.softmax(clf(tensor), dim=1)[0].cpu().numpy()

    elif clf_mode == "2class" and crop_np is not None:
        tensor = tf(image=crop_np)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            p2 = F.softmax(clf(tensor), dim=1)[0].cpu().numpy()
        p_normal, p_disease = float(p2[0]), float(p2[1])
        return np.array([p_normal, p_disease * 0.55, p_disease * 0.30, p_disease * 0.15])

    else:  # heuristic
        mp = float(prob_map[clean_mask == 1].mean()) if clean_mask.sum() > 0 else 0.0
        ds = float(np.clip(0.4 * min(lesion_pct / 30.0, 1.0) + 0.6 * mp, 0, 1))
        return np.array([1.0 - ds, ds * 0.55, ds * 0.30, ds * 0.15])


def make_overlay(image_pil, mask, alpha):
    base    = np.array(image_pil.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[mask == 1, 0] = 255
    overlay[mask == 1, 1] *= 0.15
    overlay[mask == 1, 2] *= 0.15
    blended  = (alpha * overlay + (1 - alpha) * base).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 220, 60), 2)
    return Image.fromarray(blended)


def make_gradcam(image_pil, prob_map):
    heat = cv2.cvtColor(
        cv2.applyColorMap((np.clip(prob_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    base = np.array(image_pil.convert("RGB"))
    return Image.fromarray((0.55 * heat + 0.45 * base).astype(np.uint8))


# ── Main UI ───────────────────────────────────────────────────
st.markdown("# 🦷 OralSeg — Oral Lesion Risk Assessment")
st.markdown("Upload an intraoral photograph for AI-powered lesion detection and 4-class risk classification.")

st.markdown("""
<div class="pipeline-wrap">
  <div class="pipeline-item pi-1">
    <span class="pi-dot"></span>
    <b>Stage 1</b> &nbsp;·&nbsp; SegFormer-B2 &nbsp;→&nbsp; Pixel-level segmentation
  </div>
  <div class="pipeline-item pi-2">
    <span class="pi-dot"></span>
    <b>Stage 2</b> &nbsp;·&nbsp; EfficientNet-B0 &nbsp;→&nbsp; 4-class classification
  </div>
  <div class="pipeline-item pi-3">
    <span class="pi-dot"></span>
    <b>Stage 3</b> &nbsp;·&nbsp; Normal · Ulcer · Pre-cancer · Cancer
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
seg_ok   = False
clf_mode = "none"
clf_model = None

try:
    seg_model, processor, device = load_segformer()
    seg_ok = True
except Exception as e:
    st.error(f"SegFormer failed to load: {e}")

if seg_ok:
    clf_model, clf_mode = load_classifier(device)

seg_dot = "ms-dot-ok"  if seg_ok             else "ms-dot-err"
clf_dot = "ms-dot-ok"  if clf_mode != "none" else "ms-dot-warn"
clf_lbl = {
    "4class": "Classifier (4-class) ✓",
    "2class": "Classifier (binary fallback) ✓",
    "none":   "Classifier — not loaded"
}.get(clf_mode, "")

st.markdown(f"""
<div class="model-status">
  <div class="ms-pill">
    <span class="{seg_dot}"></span>
    SegFormer {"✓" if seg_ok else "✗"}
  </div>
  <div class="ms-pill">
    <span class="{clf_dot}"></span>
    {clf_lbl}
  </div>
</div>
""", unsafe_allow_html=True)

if clf_mode == "none" and seg_ok:
    st.info("Classifier not found. Run `train_4class.py` in Colab, "
            "upload `classifier4.pth` to HuggingFace, then relaunch.", icon="ℹ️")

st.markdown("---")

uploaded = st.file_uploader(
    "Upload intraoral image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.caption("Drag and drop or browse · JPG / PNG · Max 200MB")

# ── Inference ─────────────────────────────────────────────────
if uploaded and seg_ok:
    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"Cannot read image: {e}")
        st.stop()

    orig_w, orig_h = image.size
    image_np       = np.array(image)

    with st.spinner("Running pipeline…"):
        raw_mask, prob_map = run_segmentation(image, conf_thresh, seg_model, processor, device)
        clean_mask = postprocess(raw_mask.copy(), (orig_h, orig_w),
                                 remove_small, min_blob_pct, smooth_mask)
        lesion_px    = int(clean_mask.sum())
        lesion_pct   = lesion_px / (orig_h * orig_w) * 100
        _, n_regions = ndimage.label(clean_mask)
        is_normal    = lesion_pct < NO_DISEASE_THRESHOLD_PCT
        crop_np      = crop_lesion(image_np, clean_mask) if not is_normal else None

        if is_normal:
            probs = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            probs = get_probs(crop_np, clf_model, clf_mode,
                              lesion_pct, prob_map, clean_mask, device)

        predicted_cls = 0 if is_normal else int(np.argmax(probs))
        confidence    = float(probs[predicted_cls]) * 100
        overlay_img   = make_overlay(image, clean_mask, overlay_alpha)
        gradcam_img   = make_gradcam(image, prob_map)

    # ── Banner ─────────────────────────────────────────────
    BANNERS = {
        0: ("✅ No Disease Detected — oral cavity appears healthy", "rb-normal"),
        1: (f"🟠 Benign Ulcer Detected ({confidence:.0f}% confidence)", "rb-ulcer"),
        2: (f"⚠️ Pre-cancerous Lesion Suspected ({confidence:.0f}% confidence)", "rb-precancer"),
        3: (f"🔴 High-Risk Lesion Detected ({confidence:.0f}% confidence)", "rb-cancer"),
    }
    banner_text, banner_css = BANNERS[predicted_cls]
    st.markdown(
        f'<div class="result-banner {banner_css}">{banner_text}</div>',
        unsafe_allow_html=True
    )

    # ── Tags ───────────────────────────────────────────────
    risk_map = {0: ("None", "tag-green"), 1: ("Low", "tag-orange"),
                2: ("Moderate", "tag-yellow"), 3: ("High", "tag-red")}
    risk_label, risk_tag = risk_map[predicted_cls]
    clf_tag = (
        "<span class='tag tag-purple'>4-class classifier</span>" if clf_mode == "4class" else
        "<span class='tag tag-orange'>binary classifier</span>"  if clf_mode == "2class" else
        "<span class='tag tag-gray'>heuristic estimate</span>"
    )
    st.markdown(f"""
    <div class="status-tags">
      <span class="tag {risk_tag}">Risk: {risk_label}</span>
      <span class="tag tag-gray">Coverage: {lesion_pct:.1f}%</span>
      <span class="tag tag-gray">Regions: {n_regions}</span>
      <span class="tag tag-gray">Threshold: {conf_thresh:.0%}</span>
      {clf_tag}
    </div>
    """, unsafe_allow_html=True)

    # ── Visual analysis ────────────────────────────────────
    st.markdown(
        '<div class="section-header">Visual Analysis <div class="section-line"></div></div>',
        unsafe_allow_html=True
    )
    n_cols   = 3 if show_gradcam else 2
    img_cols = st.columns(n_cols, gap="medium")

    with img_cols[0]:
        st.markdown("**Original**")
        st.image(image, use_container_width=True)
    with img_cols[1]:
        st.markdown("**Segmentation Overlay**")
        if is_normal:
            st.image(image, use_container_width=True)
            st.caption("No lesion detected")
        else:
            st.image(overlay_img, use_container_width=True)
            st.caption("🔴 Lesion  ·  🟢 Boundary")
    if show_gradcam:
        with img_cols[2]:
            st.markdown("**Attention Heatmap**")
            st.image(gradcam_img, use_container_width=True)
            st.caption("🔥 Warm = high lesion probability")

    # ── Cropped patch ──────────────────────────────────────
    if show_crop and crop_np is not None and not is_normal:
        st.markdown(
            '<div class="section-header">Stage 2 Input <div class="section-line"></div></div>',
            unsafe_allow_html=True
        )
        ci1, ci2, _ = st.columns([1, 1, 2])
        with ci1:
            st.image(cv2.resize(crop_np, (224, 224)),
                     caption="Cropped lesion (224×224)", use_container_width=True)
        with ci2:
            st.markdown(
                '<div class="metric-card" style="text-align:left; margin-top:4px">'
                '<div class="metric-label">Classifier Input</div>'
                '<div style="font-size:13px;color:#374151;margin-top:8px;line-height:1.9">'
                'EfficientNet-B0<br>↓<br><b>Normal · Ulcer · Pre-cancer · Cancer</b>'
                '</div></div>',
                unsafe_allow_html=True
            )

    # ── Metrics ────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">Measurements <div class="section-line"></div></div>',
        unsafe_allow_html=True
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    pc = "#15803d" if is_normal else "#854d0e" if lesion_pct < 5 else "#991b1b"
    for col, label, value, color in [
        (mc1, "Lesion Coverage", f"{lesion_pct:.1f}%", pc),
        (mc2, "Lesion Pixels",   f"{lesion_px:,}",     "#0f172a"),
        (mc3, "Regions Found",   str(n_regions),        "#0f172a"),
        (mc4, "Confidence",      f"{confidence:.0f}%",  CLASS_COLORS[predicted_cls]),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{color}">{value}</div></div>',
                unsafe_allow_html=True
            )

    # ── Classification breakdown ───────────────────────────
    st.markdown(
        '<div class="section-header">Classification Breakdown <div class="section-line"></div></div>',
        unsafe_allow_html=True
    )

    conf_str = f"{confidence:.0f}%"
    PRED_TEXTS = {
        0: f"Normal — No significant lesion detected ({conf_str})",
        1: f"Benign Ulcer — Likely inflammatory ({conf_str} confidence)",
        2: f"Pre-cancerous Lesion Suspected ({conf_str} confidence)",
        3: f"Cancerous Lesion — High Suspicion ({conf_str} confidence)",
    }
    st.markdown(
        f'<div class="prediction-box {CLASS_PRED[predicted_cls]}">'
        f'{CLASS_ICONS[predicted_cls]} &nbsp; {PRED_TEXTS[predicted_cls]}'
        f'</div>',
        unsafe_allow_html=True
    )

    bars_html = '<div class="risk-panel"><div class="risk-panel-title">🔬 Class Probability Breakdown</div>'
    for i in range(4):
        pct   = float(probs[i]) * 100
        badge = (f'<span class="predicted-badge {CLASS_BADGES[i]}">predicted</span>'
                 if i == predicted_cls else "")
        bars_html += (
            f'<div class="class-row">'
            f'<span class="class-label">{CLASS_ICONS[i]} {CLASS_NAMES[i]}{badge}</span>'
            f'<div class="class-bar-outer">'
            f'<div class="class-bar-inner" style="width:{pct:.1f}%;background:{CLASS_COLORS[i]}"></div>'
            f'</div>'
            f'<span class="class-pct">{pct:.1f}%</span>'
            f'</div>'
        )
    note = {
        "4class": "Probabilities from trained 4-class EfficientNet-B0 (Normal / Ulcer / Pre-cancer / Cancer).",
        "2class": "Binary classifier used. Run train_4class.py and upload classifier4.pth for real 4-class probabilities.",
        "none":   "Heuristic estimate shown. Upload classifier4.pth to HuggingFace for model-based probabilities.",
    }.get(clf_mode, "")
    bars_html += f'<div class="panel-note">ℹ️ {note}</div></div>'
    st.markdown(bars_html, unsafe_allow_html=True)

    # ── Clinical recommendation ────────────────────────────
    st.markdown(
        '<div class="section-header">Clinical Recommendation <div class="section-line"></div></div>',
        unsafe_allow_html=True
    )
    RECS = {
        0: ("✅", "success", "**Normal** — No significant lesion detected. Routine dental check-up recommended."),
        1: ("🟠", "warning", "**Benign Ulcer** — Likely aphthous, herpetic, or traumatic ulcer. Monitor for 2 weeks. If no improvement, consult a dentist."),
        2: ("⚠️", "warning", "**Pre-cancerous Lesion** — Pattern consistent with Leukoplakia or OSMF. **Clinical examination within 2 weeks is strongly recommended.**"),
        3: ("🚨", "error",   "**High-Risk Lesion** — Pattern consistent with OSCC or malignancy. **Immediate referral to an oral oncology specialist is required. Tissue biopsy is needed.**"),
    }
    icon, kind, msg = RECS[predicted_cls]
    getattr(st, kind)(msg, icon=icon)

    # ── Downloads ──────────────────────────────────────────
    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        st.download_button("⬇️ Download overlay", buf.getvalue(),
                           "lesion_overlay.png", "image/png")
    with d2:
        buf2 = io.BytesIO()
        Image.fromarray(clean_mask * 255).save(buf2, format="PNG")
        st.download_button("⬇️ Download binary mask", buf2.getvalue(),
                           "lesion_mask.png", "image/png")

    with st.expander("Raw probability heatmap (Stage 1)"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(prob_map, cmap="plasma", vmin=0, vmax=1)
        ax.set_title("P(lesion) per pixel — SegFormer output", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.03)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown(
        '<div class="disclaimer">⚕️ <strong>Medical Disclaimer:</strong> '
        'This system performs AI-based <em>screening only</em> — it is not a diagnostic device. '
        'Confidence scores are model estimates, not clinical ground truth. '
        'The presence or absence of a lesion does not confirm or exclude malignancy. '
        '<strong>Final diagnosis requires clinical examination and tissue biopsy '
        'by a qualified dental or oncology specialist.</strong></div>',
        unsafe_allow_html=True,
    )

elif not seg_ok:
    st.error("SegFormer model not loaded. Check your HuggingFace repository.")
else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#94a3b8">
      <div style="font-size:52px;margin-bottom:16px">🦷</div>
      <div style="font-size:16px;font-weight:600;color:#374151;margin-bottom:8px">
        Upload an intraoral photograph to begin
      </div>
      <div style="font-size:13px">Supported: JPG, JPEG, PNG</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Stage 1: SegFormer-B2 · Val Dice 0.6891 · Test Dice 0.6370 &nbsp;·&nbsp; "
    "Stage 2: EfficientNet-B0 · 4-Class (Normal / Ulcer / Pre-cancer / Cancer) &nbsp;·&nbsp; "
    "Based on Zhang et al., Bioengineering 2024 &nbsp;·&nbsp; "
    "⚠️ Screening tool only — not a diagnostic device"
)
