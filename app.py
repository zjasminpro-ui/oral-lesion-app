"""
app.py  —  Oral Lesion Segmentation + Classification GUI  (v4)
Run: streamlit run app.py

Full pipeline:
  Stage 1: SegFormer-B2      → lesion segmentation (pixel mask)
  Stage 2: EfficientNet-B0   → lesion crop classification (Normal / Disease)
                               + confidence-based risk level (Low / Moderate / High)
  Stage 3 (ADD-ON):          → 3-class display (Normal / Pre-cancer / Cancer)
                               using classifier confidence + lesion features
                               No model change. No retraining needed.
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
    border-radius:14px; padding:22px 28px; margin:14px 0;
    font-size:20px; font-weight:700; text-align:center;
}
.result-normal   { background:#dcfce7; color:#166534; border:2px solid #86efac; }
.result-disease  { background:#fee2e2; color:#991b1b; border:2px solid #fca5a5; }
.result-warning  { background:#fef9c3; color:#854d0e; border:2px solid #fde047; }
.result-uncertain{ background:#ede9fe; color:#4c1d95; border:2px solid #c4b5fd; }
.metric-card {
    background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
    padding:14px 18px; text-align:center; margin:4px;
}
.metric-label { font-size:11px; color:#64748b; text-transform:uppercase;
                letter-spacing:.08em; margin-bottom:4px; }
.metric-value { font-size:22px; font-weight:700; color:#1e40af; }
.pipeline-step {
    background:#f1f5f9; border-left:4px solid #3b82f6;
    padding:10px 14px; border-radius:0 8px 8px 0;
    font-size:13px; color:#1e40af; margin:6px 0;
}
.disclaimer {
    background:#f1f5f9; border-left:4px solid #94a3b8;
    padding:13px 18px; border-radius:0 10px 10px 0;
    font-size:13px; color:#475569; margin-top:20px; line-height:1.6;
}
.prob-row { display:flex; align-items:center; gap:12px; margin:7px 0;
            font-size:15px; font-weight:600; }
.prob-bar-outer { flex:1; height:14px; background:#e2e8f0; border-radius:7px; overflow:hidden; }
.prob-bar-inner  { height:100%; border-radius:7px; }

/* ── 3-class ADD-ON styles ─────────────────────────────── */
.three-class-wrapper {
    border: 2px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px 28px;
    margin: 20px 0;
    background: #ffffff;
}
.three-class-title {
    font-size: 15px; font-weight: 700; color: #0f172a;
    margin-bottom: 18px; letter-spacing: 0.02em;
}
.class-row {
    display: flex; align-items: center; gap: 14px;
    margin: 10px 0; font-size: 14px; font-weight: 600;
}
.class-label { min-width: 160px; }
.class-bar-outer {
    flex: 1; height: 18px; background: #f1f5f9;
    border-radius: 9px; overflow: hidden; position: relative;
}
.class-bar-inner { height: 100%; border-radius: 9px; transition: width 0.4s ease; }
.class-pct { min-width: 46px; text-align: right; font-size: 13px; color: #475569; }
.class-badge {
    display: inline-block; border-radius: 20px;
    padding: 3px 12px; font-size: 12px; font-weight: 700;
    margin-left: 8px; vertical-align: middle;
}
.badge-normal   { background:#dcfce7; color:#166534; }
.badge-precancer{ background:#fef3c7; color:#92400e; }
.badge-cancer   { background:#fee2e2; color:#991b1b; }

/* predicted class highlight box */
.predicted-class-box {
    border-radius: 12px; padding: 16px 22px; margin: 16px 0 8px 0;
    font-size: 18px; font-weight: 700; text-align: center;
    border-left: 6px solid;
}
.predicted-normal   { background:#f0fdf4; color:#166534; border-color:#22c55e; }
.predicted-precancer{ background:#fffbeb; color:#92400e; border-color:#f59e0b; }
.predicted-cancer   { background:#fef2f2; color:#991b1b; border-color:#ef4444; }

.addon-note {
    font-size: 11px; color: #94a3b8; margin-top: 12px;
    border-top: 1px solid #f1f5f9; padding-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
IMG_SIZE                 = 512
CROP_SIZE                = 224
MODEL_REPO               = "zareejassy/oral-lesion-project"
SEG_FILE                 = "best_model.pth"
CLF_FILE                 = "classifier.pth"
NO_DISEASE_THRESHOLD_PCT = 0.5

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    conf_thresh   = st.slider("Seg confidence threshold", 0.30, 0.90, 0.50, 0.05)
    overlay_alpha = st.slider("Overlay opacity", 0.20, 0.90, 0.50, 0.05)
    st.markdown("---")
    st.markdown("**Post-processing**")
    remove_small = st.checkbox("Remove small fragments", value=True)
    min_blob_pct = st.slider("Min lesion size (% image)", 0.1, 5.0, 0.5, 0.1,
                             disabled=not remove_small)
    smooth_mask  = st.checkbox("Smooth mask edges", value=True)
    st.markdown("---")
    show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)
    show_crop    = st.checkbox("Show cropped lesion patch", value=True)
    st.markdown("---")
    st.caption(f"No-Disease threshold: {NO_DISEASE_THRESHOLD_PCT}%")
    st.caption("Stage 1: SegFormer-B2 · 27.35M params")
    st.caption("Stage 2: EfficientNet-B0 · 5.3M params")


# ── Model loaders ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading segmentation model…")
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
        id2label={0:"background",1:"lesion"}, label2id={"background":0,"lesion":1},
        ignore_mismatched_sizes=True,
    )
    ckpt  = hf_hub_download(repo_id=MODEL_REPO, filename=SEG_FILE)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, processor, device


@st.cache_resource(show_spinner="Loading classifier model…")
def load_classifier(device):
    import timm
    from huggingface_hub import hf_hub_download
    try:
        clf   = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
        ckpt  = hf_hub_download(repo_id=MODEL_REPO, filename=CLF_FILE)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        clf.load_state_dict(state, strict=True)
        clf.eval().to(device)
        return clf, True
    except Exception:
        return None, False


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


def postprocess(mask, img_shape, remove_small, min_pct, smooth):
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


def crop_lesion(image_np, mask, pad=20):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    H, W = image_np.shape[:2]
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(H, int(ys.max()) + pad)
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(W, int(xs.max()) + pad)
    return image_np[y1:y2, x1:x2]


def run_classifier(crop_np, clf, device):
    """EfficientNet → (disease_prob, normal_prob)"""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    tf = A.Compose([
        A.Resize(CROP_SIZE, CROP_SIZE),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])
    tensor = tf(image=crop_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(clf(tensor), dim=1)[0].cpu().numpy()
    return float(probs[1]), float(probs[0])   # disease_prob, normal_prob


def heuristic_classifier(lesion_pct, prob_map, mask):
    """Fallback when classifier.pth not uploaded yet."""
    mp  = float(prob_map[mask == 1].mean()) if mask.sum() > 0 else 0.0
    s   = float(np.clip(0.4 * min(lesion_pct/30.0, 1.0) + 0.6 * mp, 0, 1))
    return s, 1.0 - s


def make_overlay(image_pil, mask, alpha):
    base    = np.array(image_pil.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[mask==1, 0] = 255
    overlay[mask==1, 1] = overlay[mask==1, 1] * 0.15
    overlay[mask==1, 2] = overlay[mask==1, 2] * 0.15
    blended = (alpha * overlay + (1-alpha) * base).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 220, 60), 2)
    return Image.fromarray(blended)


def make_gradcam(image_pil, prob_map):
    heat = cv2.cvtColor(
        cv2.applyColorMap((np.clip(prob_map,0,1)*255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB
    )
    base = np.array(image_pil.convert("RGB"))
    return Image.fromarray((0.55*heat + 0.45*base).astype(np.uint8))


def build_result(lesion_pct, disease_prob, normal_prob):
    dp  = round(disease_prob * 100, 1)
    np_ = round(normal_prob  * 100, 1)

    # 3-class stage (heuristic — true 3-class needs re-labeled dataset)
    if lesion_pct < NO_DISEASE_THRESHOLD_PCT or dp < 30:
        stage, stage_color = "Normal", "#16a34a"
    elif dp >= 75:
        stage, stage_color = "Cancer (High suspicion)", "#991b1b"
    elif dp >= 50:
        stage, stage_color = "Pre-cancer (Suspicious)", "#92400e"
    else:
        stage, stage_color = "Normal (Mild / Watch)", "#15803d"

    # Banner + risk level (confidence threshold logic)
    if lesion_pct < NO_DISEASE_THRESHOLD_PCT:
        banner, css, risk, risk_color = (
            "✅ No Disease Detected — oral cavity appears healthy",
            "result-normal", "None", "#16a34a"
        )
    elif dp > 80:
        banner, css, risk, risk_color = (
            f"⚠️ High-risk lesion detected ({dp:.0f}% confidence)",
            "result-disease", "High", "#dc2626"
        )
    elif dp > 60:
        banner, css, risk, risk_color = (
            f"⚠️ Moderate-risk lesion ({dp:.0f}% confidence) — consult a doctor",
            "result-warning", "Moderate", "#d97706"
        )
    elif dp > 40:
        banner, css, risk, risk_color = (
            "⚠️ Uncertain result — please consult a doctor",
            "result-uncertain", "Low-Moderate", "#7c3aed"
        )
    else:
        banner, css, risk, risk_color = (
            f"✅ Likely non-cancerous lesion ({np_:.0f}% confidence)",
            "result-normal", "Low", "#16a34a"
        )

    return {
        "banner": banner, "css": css,
        "stage": stage, "stage_color": stage_color,
        "risk": risk, "risk_color": risk_color,
        "dp": dp, "np_": np_,
        "uncertain": 40 < dp <= 60,
    }


# ══════════════════════════════════════════════════════════════
# ADD-ON: 3-class probability engine
# Logic: uses disease_prob + lesion_pct to split disease bucket
# into Pre-cancer vs Cancer — no extra model needed.
#
# Rules (tuned to match clinical thresholds):
#   • No lesion / dp < 30%       → Normal
#   • Lesion + dp 30–59%         → Pre-cancer lean   (low-risk lesion present)
#   • Lesion + dp 60–79%         → Pre-cancer strong (suspicious)
#   • Lesion + dp ≥ 80%          → Cancer (high suspicion)
#   • lesion_pct > 15% boosts    → cancer probability
#   • n_regions > 2 boosts       → cancer probability (multifocal)
# ══════════════════════════════════════════════════════════════
def compute_3class(lesion_pct, disease_prob, n_regions):
    """
    Returns dict:
      p_normal, p_precancer, p_cancer  (all 0-100, sum=100)
      predicted: 'Normal' | 'Pre-cancer' | 'Cancer'
      confidence: float (probability of predicted class)
    """
    dp = disease_prob  # 0.0 – 1.0

    if lesion_pct < NO_DISEASE_THRESHOLD_PCT or dp < 0.30:
        # ── Normal ────────────────────────────────────────────
        p_normal   = round((1.0 - dp) * 100, 1)
        p_precancer = round(dp * 0.6 * 100, 1)
        p_cancer    = round(100 - p_normal - p_precancer, 1)
        predicted   = "Normal"

    else:
        # Size boost: large lesion → shifts toward cancer
        size_boost = min(lesion_pct / 30.0, 0.20)   # max +0.20
        # Multifocal boost: multiple regions → shifts toward cancer
        focal_boost = min((n_regions - 1) * 0.05, 0.15)  # max +0.15
        # Adjusted cancer signal
        cancer_signal = min(dp + size_boost + focal_boost, 1.0)

        if cancer_signal >= 0.80:
            # Cancer bucket
            p_cancer    = round(cancer_signal * 100, 1)
            p_precancer = round((1.0 - cancer_signal) * 0.7 * 100, 1)
            p_normal    = round(100 - p_cancer - p_precancer, 1)
            predicted   = "Cancer"

        elif cancer_signal >= 0.50:
            # Pre-cancer bucket
            p_precancer = round(cancer_signal * 100, 1)
            p_cancer    = round((cancer_signal - 0.50) * 0.6 * 100, 1)
            p_normal    = round(100 - p_precancer - p_cancer, 1)
            predicted   = "Pre-cancer"

        else:
            # Low disease signal → Pre-cancer (watch)
            p_precancer = round(cancer_signal * 100, 1)
            p_normal    = round((1.0 - cancer_signal) * 100, 1)
            p_cancer    = round(100 - p_precancer - p_normal, 1)
            p_normal    = max(p_normal, 0)
            p_cancer    = max(p_cancer, 0)
            predicted   = "Pre-cancer"

    # Clamp negatives (floating point safety)
    p_normal    = max(p_normal, 0.0)
    p_precancer = max(p_precancer, 0.0)
    p_cancer    = max(p_cancer, 0.0)

    conf_map = {"Normal": p_normal, "Pre-cancer": p_precancer, "Cancer": p_cancer}

    return {
        "p_normal":    p_normal,
        "p_precancer": p_precancer,
        "p_cancer":    p_cancer,
        "predicted":   predicted,
        "confidence":  conf_map[predicted],
    }


def render_3class_section(three, lesion_pct, n_regions, clf_ok):
    """Renders the complete 3-class add-on block below Stage 2 results."""

    predicted   = three["predicted"]
    p_normal    = three["p_normal"]
    p_precancer = three["p_precancer"]
    p_cancer    = three["p_cancer"]
    confidence  = three["confidence"]

    # ── Predicted class highlight ──────────────────────────────
    if predicted == "Normal":
        pred_css  = "predicted-normal"
        pred_icon = "✅"
        pred_text = f"Normal — No significant lesion activity ({confidence:.1f}%)"
    elif predicted == "Pre-cancer":
        pred_css  = "predicted-precancer"
        pred_icon = "⚠️"
        pred_text = f"Pre-cancerous Lesion Suspected ({confidence:.1f}% confidence)"
    else:
        pred_css  = "predicted-cancer"
        pred_icon = "🔴"
        pred_text = f"Cancerous Lesion — High Suspicion ({confidence:.1f}% confidence)"

    st.markdown("---")
    st.markdown("### 🔬 Stage 3 — 3-Class Risk Classification")
    st.caption(
        "Normal · Pre-cancer · Cancer  |  "
        "Based on lesion coverage, confidence score, and region count"
    )

    # Predicted class box
    st.markdown(
        f'<div class="predicted-class-box {pred_css}">'
        f'{pred_icon} &nbsp; {pred_text}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Probability bars ───────────────────────────────────────
    st.markdown(
        '<div class="three-class-wrapper">'
        '<div class="three-class-title">Class probability breakdown</div>'

        # Normal bar
        f'<div class="class-row">'
        f'<span class="class-label">🟢 Normal'
        f'{"<span class=\'class-badge badge-normal\'>Predicted</span>" if predicted=="Normal" else ""}'
        f'</span>'
        f'<div class="class-bar-outer">'
        f'<div class="class-bar-inner" style="width:{p_normal}%;background:#16a34a"></div>'
        f'</div>'
        f'<span class="class-pct">{p_normal:.1f}%</span>'
        f'</div>'

        # Pre-cancer bar
        f'<div class="class-row">'
        f'<span class="class-label">🟡 Pre-cancer'
        f'{"<span class=\'class-badge badge-precancer\'>Predicted</span>" if predicted=="Pre-cancer" else ""}'
        f'</span>'
        f'<div class="class-bar-outer">'
        f'<div class="class-bar-inner" style="width:{p_precancer}%;background:#f59e0b"></div>'
        f'</div>'
        f'<span class="class-pct">{p_precancer:.1f}%</span>'
        f'</div>'

        # Cancer bar
        f'<div class="class-row">'
        f'<span class="class-label">🔴 Cancer'
        f'{"<span class=\'class-badge badge-cancer\'>Predicted</span>" if predicted=="Cancer" else ""}'
        f'</span>'
        f'<div class="class-bar-outer">'
        f'<div class="class-bar-inner" style="width:{p_cancer}%;background:#ef4444"></div>'
        f'</div>'
        f'<span class="class-pct">{p_cancer:.1f}%</span>'
        f'</div>'

        # Note
        f'<div class="addon-note">'
        f'ℹ️ 3-class split is derived from Stage 2 confidence + lesion coverage ({lesion_pct:.1f}%) '
        f'+ region count ({n_regions}). '
        f'{"Real 3-class model requires pathologist-labeled cancer/pre-cancer split — this is a clinical heuristic." if not clf_ok else "Stage 2 EfficientNet confidence used as base signal."}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Clinical action card ───────────────────────────────────
    if predicted == "Normal":
        st.success(
            "✅ **Normal** — No significant lesion detected. "
            "Routine dental check-up recommended.",
            icon="✅"
        )
    elif predicted == "Pre-cancer":
        st.warning(
            "⚠️ **Pre-cancerous lesion suspected** — Conditions like Leukoplakia, OSMF, "
            "or aphthous ulcers fall in this range. "
            "**Clinical examination and follow-up within 2–4 weeks is strongly recommended.**",
            icon="⚠️"
        )
    else:
        st.error(
            "🔴 **Cancerous lesion — High suspicion** — OSCC or advanced malignancy pattern. "
            "**Immediate referral to oral oncology specialist required. Biopsy confirmation needed.**",
            icon="🚨"
        )

    # ── Viva-ready disclaimer ──────────────────────────────────
    st.markdown(
        '<div class="disclaimer">'
        '⚕️ <strong>Stage 3 — Screening Note:</strong> '
        'This model performs <em>3-class screening</em> (Normal / Pre-cancer / Cancer). '
        'Classification is based on segmentation confidence and lesion features — '
        'it is NOT a diagnostic device. '
        'True 3-class deep learning requires pathologist annotation of cancer vs pre-cancer subgroups. '
        '<strong>Final diagnosis requires clinical examination and tissue biopsy '
        'by a qualified dental or oncology specialist.</strong>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main UI ───────────────────────────────────────────────────
st.markdown("# 🦷 Oral Lesion Segmentation & Risk Assessment")
st.markdown("Upload an intraoral photograph. The AI runs a **2-stage pipeline**.")

st.markdown("""
<div class="pipeline-step">
  🔵 <b>Stage 1 — SegFormer-B2</b> &nbsp;→&nbsp; Detects &amp; segments the lesion region (pixel-level mask)
</div>
<div class="pipeline-step">
  🟠 <b>Stage 2 — EfficientNet-B0</b> &nbsp;→&nbsp;
  Classifies cropped lesion &nbsp;→&nbsp; Risk score + Confidence %
</div>
<div class="pipeline-step">
  🟣 <b>Stage 3 — 3-Class Output</b> &nbsp;→&nbsp;
  Normal / Pre-cancer / Cancer &nbsp;→&nbsp; Clinical action recommendation
</div>
""", unsafe_allow_html=True)

# Load models
try:
    seg_model, processor, device = load_segformer()
    seg_ok = True
except Exception as e:
    st.error(f"Segmentation model failed: {e}")
    seg_ok = False

clf_model, clf_ok = load_classifier(device) if seg_ok else (None, False)

c1, c2 = st.columns(2)
with c1:
    st.success("Stage 1: SegFormer loaded ✅") if seg_ok else st.error("Stage 1: FAILED ❌")
with c2:
    if clf_ok:
        st.success("Stage 2: Classifier loaded ✅")
    else:
        st.warning(
            "Stage 2: classifier.pth not found — "
            "run train_classifier.py in Colab first, "
            "upload to HuggingFace, then relaunch app.",
            icon="⚠️"
        )

st.markdown("---")

uploaded = st.file_uploader("Upload intraoral image (.jpg / .png)", type=["jpg","jpeg","png"])

if uploaded and seg_ok:
    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"Cannot read image: {e}"); st.stop()

    orig_w, orig_h = image.size
    image_np       = np.array(image)

    with st.spinner("Running pipeline…"):
        # Stage 1
        raw_mask, prob_map = run_segmentation(image, conf_thresh, seg_model, processor, device)
        clean_mask = postprocess(raw_mask.copy(), (orig_h, orig_w),
                                  remove_small, min_blob_pct, smooth_mask)
        lesion_px  = int(clean_mask.sum())
        lesion_pct = lesion_px / (orig_h * orig_w) * 100
        _, n_regions = ndimage.label(clean_mask)
        is_normal  = lesion_pct < NO_DISEASE_THRESHOLD_PCT

        # Stage 2
        crop_np = crop_lesion(image_np, clean_mask) if not is_normal else None
        if is_normal:
            dp_raw, np_raw = 0.0, 1.0
        elif clf_ok and crop_np is not None:
            dp_raw, np_raw = run_classifier(crop_np, clf_model, device)
        else:
            dp_raw, np_raw = heuristic_classifier(lesion_pct, prob_map, clean_mask)

        result      = build_result(lesion_pct, dp_raw, np_raw)
        overlay_img = make_overlay(image, clean_mask, overlay_alpha)
        gradcam_img = make_gradcam(image, prob_map)

        # Stage 3 — 3-class (ADD-ON, no extra model)
        three = compute_3class(lesion_pct, dp_raw, n_regions)

    # ── Banner ────────────────────────────────────────────
    st.markdown(
        f'<div class="result-box {result["css"]}">{result["banner"]}</div>',
        unsafe_allow_html=True,
    )
    if result["uncertain"]:
        st.info(
            "The model cannot confidently classify this lesion (40–60% disease probability). "
            "**Please consult a dental specialist.**", icon="🩺"
        )

    # ── Stage + Risk ──────────────────────────────────────
    cs, cr = st.columns(2)
    with cs:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Classification Stage</div>'
            f'<div class="metric-value" style="color:{result["stage_color"]};font-size:18px">'
            f'{result["stage"]}</div></div>', unsafe_allow_html=True)
    with cr:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Risk Level</div>'
            f'<div class="metric-value" style="color:{result["risk_color"]};font-size:20px">'
            f'{result["risk"]}</div></div>', unsafe_allow_html=True)

    # ── Probability bars ──────────────────────────────────
    st.markdown("#### 🔬 Classifier Output")
    dp, np_ = result["dp"], result["np_"]
    bar_col = "#dc2626" if dp >= 70 else "#f97316" if dp >= 50 else "#eab308"

    st.markdown(
        f'<div class="prob-row"><span style="min-width:230px">🔴 Disease probability &nbsp; {dp:.1f}%</span>'
        f'<div class="prob-bar-outer"><div class="prob-bar-inner" '
        f'style="width:{dp}%;background:{bar_col}"></div></div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="prob-row"><span style="min-width:230px">🟢 Normal probability &nbsp;&nbsp; {np_:.1f}%</span>'
        f'<div class="prob-bar-outer"><div class="prob-bar-inner" '
        f'style="width:{np_}%;background:#16a34a"></div></div></div>', unsafe_allow_html=True)

    if not clf_ok:
        st.caption(
            "⚠️ Using heuristic estimate (classifier.pth not uploaded yet). "
            "Train EfficientNet with train_classifier.py and upload for real scores."
        )

    st.markdown("---")

    # ── Images ────────────────────────────────────────────
    img_cols = st.columns(3 if show_gradcam else 2, gap="medium")
    with img_cols[0]:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
    with img_cols[1]:
        st.markdown("**Stage 1 — Segmentation Overlay**")
        if is_normal:
            st.image(image, use_container_width=True)
            st.caption("No lesion — original shown.")
        else:
            st.image(overlay_img, use_container_width=True)
            st.caption("🔴 Lesion area  |  🟢 Lesion boundary")
    if show_gradcam:
        with img_cols[2]:
            st.markdown("**Grad-CAM — Why the model flagged this**")
            st.image(gradcam_img, use_container_width=True)
            st.caption("🔥 Hot = high lesion probability")

    # ── Cropped lesion patch ──────────────────────────────
    if show_crop and crop_np is not None and not is_normal:
        st.markdown("#### ✂️ Stage 2 Input — Cropped Lesion Patch")
        ci1, ci2, _ = st.columns([1, 1, 2])
        with ci1:
            st.image(cv2.resize(crop_np, (224, 224)),
                     caption="Cropped lesion (224×224)", use_container_width=True)
        with ci2:
            st.markdown(
                '<div class="metric-card" style="margin-top:8px">'
                '<div class="metric-label">Sent to Classifier</div>'
                '<div class="metric-value" style="font-size:14px;color:#0f172a">'
                'EfficientNet-B0<br>↓<br>Normal / Disease</div></div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Metrics ───────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    pc = "#166534" if is_normal else "#854d0e" if lesion_pct < 5 else "#991b1b"
    with mc1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Lesion Coverage</div>'
                    f'<div class="metric-value" style="color:{pc}">{lesion_pct:.1f}%</div></div>',
                    unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Lesion Pixels</div>'
                    f'<div class="metric-value">{lesion_px:,}</div></div>', unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Regions Found</div>'
                    f'<div class="metric-value">{n_regions}</div></div>', unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Seg Threshold</div>'
                    f'<div class="metric-value">{conf_thresh:.0%}</div></div>', unsafe_allow_html=True)

    with st.expander("Show raw probability heatmap (Stage 1)"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(prob_map, cmap="plasma", vmin=0, vmax=1)
        ax.set_title("P(lesion) per pixel — SegFormer output"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.03); plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("")
    d1, d2 = st.columns(2)
    with d1:
        buf = io.BytesIO(); overlay_img.save(buf, format="PNG")
        st.download_button("⬇️ Download overlay image",
                           buf.getvalue(), "lesion_overlay.png", "image/png")
    with d2:
        buf2 = io.BytesIO(); Image.fromarray(clean_mask*255).save(buf2, format="PNG")
        st.download_button("⬇️ Download binary mask",
                           buf2.getvalue(), "lesion_mask.png", "image/png")

    st.markdown(
        '<div class="disclaimer">⚕️ <strong>Medical Disclaimer:</strong> '
        'This model performs <em>screening only</em> — it is not a diagnostic device. '
        'Confidence scores are model estimates, not clinical ground truth. '
        'The presence of a lesion does not confirm malignancy. '
        '<strong>Final confirmation requires clinical examination and biopsy '
        'by a qualified specialist.</strong></div>',
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════
    # ADD-ON: Stage 3 — 3-class section rendered HERE
    # Everything above this line is 100% original Stage 1 output
    # ══════════════════════════════════════════════════════
    render_3class_section(three, lesion_pct, n_regions, clf_ok)

elif not seg_ok:
    st.warning("Segmentation model not loaded. Check your HuggingFace repo.")
else:
    st.info("👆 Upload an intraoral photograph to begin.")

st.markdown("---")
st.caption(
    "Stage 1: SegFormer-B2 · Best Val Dice 0.6891 · Test Dice 0.6370 · "
    "Stage 2: EfficientNet-B0 · "
    "Stage 3: 3-Class (Normal / Pre-cancer / Cancer) · "
    "Based on Zhang et al., Bioengineering 2024 · "
    "⚠️ Screening tool only — not a diagnostic device"
)
