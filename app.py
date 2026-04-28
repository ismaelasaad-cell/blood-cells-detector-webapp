"""
Blood Cell Detector — Streamlit Web Application
=================================================
Upload a peripheral blood smear image and get instant AI-powered
cell detection and classification using a YOLO26-based model.

Detects 7 cell types:
  RBC, Platelets, Neutrophil, Lymphocyte, Monocyte, Eosinophil, Basophil
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ── Paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "blood_detector_model.pt"
METADATA_PATH = ROOT / "blood_detector_metadata.json"
TEST_IMAGES_DIR = ROOT / "test_images"

# ── Class info ──────────────────────────────────────────────────────────
CLASS_NAMES = ["RBC", "Platelets", "Neutrophil", "Lymphocyte",
               "Monocyte", "Eosinophil", "Basophil"]

WBC_SUBTYPES = {"Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"}

# Colors for drawing (BGR for OpenCV)
COLOR_MAP_BGR = {
    "RBC":        (86,  87, 232),   # warm red
    "Platelets":  (80, 200, 100),   # green
    "Neutrophil": (230, 150, 50),   # blue
    "Lymphocyte": (200, 100, 220),  # purple
    "Monocyte":   (100, 200, 230),  # gold/yellow
    "Eosinophil": (80,  130, 230),  # orange
    "Basophil":   (180, 100, 140),  # teal
}

# Colors for UI display (CSS hex)
COLOR_MAP_HEX = {
    "RBC":        "#E85756",
    "Platelets":  "#64C850",
    "Neutrophil": "#3296E6",
    "Lymphocyte": "#DC64C8",
    "Monocyte":   "#E6C864",
    "Eosinophil": "#E68250",
    "Basophil":   "#8C64B4",
}

# Emoji for cell types
CELL_EMOJI = {
    "RBC":        "🔴",
    "Platelets":  "🟡",
    "Neutrophil": "🔵",
    "Lymphocyte": "🟣",
    "Monocyte":   "🟠",
    "Eosinophil": "🟤",
    "Basophil":   "⚫",
}


# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Blood Cell Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ──────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Import fonts ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ───────────────────────────────────────────── */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(145deg, #0a0e1a 0%, #111827 50%, #0f172a 100%);
    }

    /* ── Header ───────────────────────────────────────────── */
    .hero-header {
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
        margin-bottom: 1rem;
    }
    .hero-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
    }

    /* ── Glass cards ──────────────────────────────────────── */
    .glass-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.3);
        box-shadow: 0 8px 32px rgba(96, 165, 250, 0.1);
    }
    .glass-card h3 {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }

    /* ── Stat pill ────────────────────────────────────────── */
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(100, 116, 139, 0.25);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin: 0.25rem;
        transition: all 0.25s ease;
    }
    .stat-pill:hover {
        transform: translateY(-2px);
        border-color: rgba(96, 165, 250, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .stat-pill .count {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .stat-pill .label {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .stat-pill .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Upload area ──────────────────────────────────────── */
    .upload-zone {
        border: 2px dashed rgba(96, 165, 250, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(30, 41, 59, 0.3);
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: rgba(96, 165, 250, 0.6);
        background: rgba(30, 41, 59, 0.5);
    }

    /* ── Results summary row ──────────────────────────────── */
    .summary-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        justify-content: center;
        margin: 1rem 0;
    }

    /* ── Sidebar ──────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(100, 116, 139, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    /* ── Model info badge ─────────────────────────────────── */
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.15), rgba(167, 139, 250, 0.15));
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.75rem;
        color: #93c5fd;
        font-weight: 500;
    }

    /* ── WBC diff table ───────────────────────────────────── */
    .wbc-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top: 0.5rem;
    }
    .wbc-table th {
        background: rgba(30, 41, 59, 0.8);
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 0.6rem 0.8rem;
        border-bottom: 1px solid rgba(100, 116, 139, 0.2);
        text-align: left;
    }
    .wbc-table td {
        padding: 0.55rem 0.8rem;
        color: #e2e8f0;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(100, 116, 139, 0.1);
    }
    .wbc-table tr:hover td {
        background: rgba(96, 165, 250, 0.05);
    }
    .wbc-bar {
        height: 6px;
        border-radius: 3px;
        transition: width 0.6s ease;
    }

    /* ── Disclaimer ───────────────────────────────────────── */
    .disclaimer {
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.2);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        color: #fbbf24;
        font-size: 0.8rem;
        margin-top: 1rem;
    }

    /* ── Hide Streamlit branding (optional) ────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Fix img display ──────────────────────────────────── */
    .stImage > img {
        border-radius: 12px;
        border: 1px solid rgba(100, 116, 139, 0.2);
    }

    /* ── Example image button ─────────────────────────────── */
    .example-btn {
        display: block;
        width: 100%;
        padding: 0.5rem;
        margin: 0.3rem 0;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(100, 116, 139, 0.2);
        border-radius: 8px;
        color: #cbd5e1;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
        text-align: left;
    }
    .example-btn:hover {
        background: rgba(96, 165, 250, 0.1);
        border-color: rgba(96, 165, 250, 0.3);
    }

    /* ── Confidence bar animation ─────────────────────────── */
    @keyframes fillBar {
        from { width: 0%; }
    }
    .wbc-bar {
        animation: fillBar 0.8s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)


# ── Model loading (cached) ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load YOLO model once and cache it across reruns."""
    return YOLO(str(MODEL_PATH))


def load_metadata() -> dict:
    """Load model metadata JSON."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH) as f:
            return json.load(f)
    return {}


# ── Detection logic ─────────────────────────────────────────────────────
def run_detection(model: YOLO, image: np.ndarray,
                  conf: float, iou: float) -> dict:
    """Run YOLO inference and return structured results."""
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=640,
        device="cpu",
        save=False,
        verbose=False,
        max_det=300,
    )
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    names = [r.names[c] for c in classes]

    return {
        "boxes": boxes,
        "classes": classes,
        "confidences": confs,
        "names": names,
        "class_map": r.names,
    }


def annotate_image(image: np.ndarray, det: dict) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.45, 1

    for (x1, y1, x2, y2), name, conf in zip(
            det["boxes"], det["names"], det["confidences"]):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        col = COLOR_MAP_BGR.get(name, (200, 200, 200))

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

        # Label with confidence
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
        ly = y1 - 2
        if ly - th - 4 < 0:
            ly = y1 + th + 6

        # Label background
        cv2.rectangle(img, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), col, -1)
        cv2.putText(img, label, (x1 + 2, ly - 1), font, fs,
                    (255, 255, 255), ft, cv2.LINE_AA)

    return img


# ── UI helpers ───────────────────────────────────────────────────────────
def render_stat_pills(counts: Counter, total: int):
    """Render cell count summary as coloured pill badges."""
    pills_html = ""
    for cls_name in CLASS_NAMES:
        count = counts.get(cls_name, 0)
        if count == 0:
            continue
        color = COLOR_MAP_HEX.get(cls_name, "#94a3b8")
        emoji = CELL_EMOJI.get(cls_name, "")
        pills_html += f"""
        <div class="stat-pill">
            <span class="dot" style="background: {color};"></span>
            <span class="count">{count}</span>
            <span class="label">{emoji} {cls_name}</span>
        </div>
        """
    st.markdown(f'<div class="summary-grid">{pills_html}</div>',
                unsafe_allow_html=True)


def render_wbc_differential(counts: Counter):
    """Render WBC differential table with horizontal bars."""
    wbc_counts = {k: counts.get(k, 0) for k in WBC_SUBTYPES}
    wbc_total = sum(wbc_counts.values())
    if wbc_total == 0:
        st.info("No WBCs detected to compute differential.")
        return

    rows = ""
    for name in ["Neutrophil", "Lymphocyte", "Monocyte", "Eosinophil", "Basophil"]:
        c = wbc_counts[name]
        pct = (c / wbc_total * 100) if wbc_total > 0 else 0
        color = COLOR_MAP_HEX.get(name, "#94a3b8")
        emoji = CELL_EMOJI.get(name, "")
        rows += f"""
        <tr>
            <td>{emoji} {name}</td>
            <td style="font-weight:600;">{c}</td>
            <td style="font-weight:600;">{pct:.1f}%</td>
            <td style="width:40%;">
                <div style="background:rgba(100,116,139,0.15);border-radius:3px;overflow:hidden;">
                    <div class="wbc-bar" style="width:{pct}%;background:{color};"></div>
                </div>
            </td>
        </tr>
        """

    st.markdown(f"""
    <table class="wbc-table">
        <thead>
            <tr><th>Cell Type</th><th>Count</th><th>%</th><th>Distribution</th></tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ── Main application ────────────────────────────────────────────────────
def main():
    inject_css()

    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>🔬 Blood Cell Detector</h1>
        <p>AI-powered peripheral blood smear analysis — upload an image to detect and classify cells</p>
        <div style="margin-top:0.6rem;">
            <span class="model-badge">⚡ YOLO26 &nbsp;·&nbsp; 7 classes &nbsp;·&nbsp; 640×640</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Detection Settings")

        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05, max_value=0.95, value=0.25, step=0.05,
            help="Minimum confidence score to keep a detection. "
                 "Lower = more detections (may include false positives). "
                 "Higher = fewer but more confident detections.",
        )

        iou_threshold = st.slider(
            "IoU threshold (NMS)",
            min_value=0.1, max_value=0.9, value=0.7, step=0.05,
            help="Intersection-over-Union threshold for Non-Maximum Suppression. "
                 "Higher = allows more overlapping boxes.",
        )

        st.markdown("---")
        st.markdown("## 🎯 Cell Types Detected")
        for cls_name in CLASS_NAMES:
            color = COLOR_MAP_HEX[cls_name]
            emoji = CELL_EMOJI[cls_name]
            st.markdown(
                f'<span style="color:{color};font-weight:600;">'
                f'{emoji} {cls_name}</span>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("## 📂 Try Example Images")

        # List example images
        example_images = sorted(
            p for p in TEST_IMAGES_DIR.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ) if TEST_IMAGES_DIR.exists() else []

        selected_example = None
        for img_path in example_images:
            nice_name = img_path.stem.replace("_", " ").title()
            if st.button(f"🖼️ {nice_name}", key=f"ex_{img_path.stem}",
                         use_container_width=True):
                selected_example = img_path

        st.markdown("---")

        # Model metadata
        meta = load_metadata()
        if meta:
            st.markdown("## 📋 Model Info")
            st.markdown(f"""
            - **Architecture:** YOLO26m  
            - **Classes:** {meta.get('nc', 7)}  
            - **Input size:** {meta.get('imgsz', 640)}×{meta.get('imgsz', 640)}  
            - **Ultralytics:** v{meta.get('ultralytics_version_trained', 'N/A')}
            """)

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Research use only.</strong> Not validated as a medical device. 
            Do not use for clinical diagnosis or treatment decisions.
        </div>
        """, unsafe_allow_html=True)

    # ── Main content ────────────────────────────────────────
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a blood smear image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF",
        label_visibility="collapsed",
    )

    # Determine which image to use
    image_source = None
    source_label = ""

    if uploaded_file is not None:
        image_source = uploaded_file
        source_label = uploaded_file.name
    elif selected_example is not None:
        st.session_state["selected_example"] = str(selected_example)
        image_source = selected_example
        source_label = selected_example.name
    elif "selected_example" in st.session_state:
        path = Path(st.session_state["selected_example"])
        if path.exists():
            image_source = path
            source_label = path.name

    if image_source is None:
        # Show placeholder
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:3rem;margin-bottom:0.5rem;">📤</div>
            <div style="color:#94a3b8;font-size:1.1rem;font-weight:500;">
                Drop a blood smear image here or click to upload
            </div>
            <div style="color:#64748b;font-size:0.85rem;margin-top:0.3rem;">
                Or select an example image from the sidebar →
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Load image ──────────────────────────────────────────
    if isinstance(image_source, Path):
        pil_image = Image.open(image_source)
    else:
        pil_image = Image.open(image_source)

    # Convert to RGB numpy array for YOLO
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ── Run detection ───────────────────────────────────────
    with st.spinner("🔍 Analyzing blood smear..."):
        model = load_model()
        det = run_detection(model, img_bgr, conf_threshold, iou_threshold)

    # ── Annotate image ──────────────────────────────────────
    annotated_bgr = annotate_image(img_bgr, det)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ── Counts ──────────────────────────────────────────────
    counts = Counter(det["names"])
    total = sum(counts.values())

    # ── Source info ──────────────────────────────────────────
    st.markdown(
        f'<p style="color:#64748b;font-size:0.85rem;text-align:center;">'
        f'📄 <strong>{source_label}</strong> &nbsp;·&nbsp; '
        f'{img_rgb.shape[1]}×{img_rgb.shape[0]} px &nbsp;·&nbsp; '
        f'<strong>{total}</strong> detections</p>',
        unsafe_allow_html=True,
    )

    # ── Summary pills ──────────────────────────────────────
    render_stat_pills(counts, total)

    # ── Side-by-side images ─────────────────────────────────
    col_orig, col_det = st.columns(2)
    with col_orig:
        st.markdown('<div class="glass-card"><h3>📷 Original</h3></div>',
                    unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    with col_det:
        st.markdown('<div class="glass-card"><h3>🎯 Detections</h3></div>',
                    unsafe_allow_html=True)
        st.image(annotated_rgb, use_container_width=True)

    # ── Detailed results ────────────────────────────────────
    st.markdown("---")

    col_wbc, col_all = st.columns([1, 1])

    with col_wbc:
        st.markdown(
            '<div class="glass-card"><h3>🧬 WBC Differential</h3></div>',
            unsafe_allow_html=True,
        )
        render_wbc_differential(counts)

    with col_all:
        st.markdown(
            '<div class="glass-card"><h3>📊 All Cell Counts</h3></div>',
            unsafe_allow_html=True,
        )

        # Horizontal bar chart using Streamlit's native chart
        import pandas as pd
        chart_data = []
        for cls_name in CLASS_NAMES:
            c = counts.get(cls_name, 0)
            if c > 0:
                chart_data.append({"Cell Type": cls_name, "Count": c})

        if chart_data:
            df = pd.DataFrame(chart_data)
            st.bar_chart(df, x="Cell Type", y="Count",
                         color="#60a5fa", horizontal=True)

    # ── Detection details (expandable) ──────────────────────
    with st.expander("📋 Detection Details (per-box)"):
        if len(det["boxes"]) > 0:
            import pandas as pd
            detail_rows = []
            for i, (box, name, conf) in enumerate(zip(
                    det["boxes"], det["names"], det["confidences"])):
                x1, y1, x2, y2 = map(int, box)
                detail_rows.append({
                    "#": i + 1,
                    "Class": name,
                    "Confidence": f"{conf:.1%}",
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "Width": x2 - x1,
                    "Height": y2 - y1,
                })
            df_detail = pd.DataFrame(detail_rows)
            st.dataframe(df_detail, use_container_width=True, hide_index=True)
        else:
            st.info("No cells detected with current settings. "
                    "Try lowering the confidence threshold.")


if __name__ == "__main__":
    main()
