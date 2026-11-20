"""MediScan AI - clinical imaging dashboard.   Run:  streamlit run app.py"""
import base64
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image 

from src import config
from src.gradcam import overlay_cam
from src.predict import analyze, load_checkpoint
from src.quality import assess_quality
from src.report import build_pdf

st.set_page_config(page_title="MediScan AI | Clinical Imaging Assistant", page_icon="🩻",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family:'Inter',-apple-system,'Segoe UI',sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"], .stDeployButton { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:1.4rem; max-width:1280px; }
section[data-testid="stSidebar"] { background:#0A2540; }
section[data-testid="stSidebar"] * { color:#E8F0FA; }
section[data-testid="stSidebar"] .stSlider label, section[data-testid="stSidebar"] p { color:#C9DDF2 !important; }
.brand { display:flex; align-items:center; gap:12px; margin:4px 0 18px; }
.brand-logo { width:42px; height:42px; border-radius:12px; background:linear-gradient(135deg,#12A5A5,#0B5CAD);
  display:flex; align-items:center; justify-content:center; font-size:22px; }
.brand-name { font-size:1.2rem; font-weight:700; color:#fff; letter-spacing:-.3px; line-height:1.1; }
.brand-sub { font-size:.72rem; color:#8FB0D1; letter-spacing:.06em; text-transform:uppercase; }
.side-box { background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); border-radius:12px;
  padding:12px 14px; margin:10px 0; font-size:.82rem; line-height:1.5; }
.side-box b { color:#fff; }
.dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#2ED17F; margin-right:7px; }
.hero { background:linear-gradient(120deg,#0A2540 0%,#0B5CAD 55%,#12A5A5 100%); border-radius:18px;
  padding:28px 34px; margin-bottom:20px; box-shadow:0 10px 30px rgba(10,37,64,.18); }
.hero-title { font-size:2rem; font-weight:700; letter-spacing:-.5px; color:#fff; line-height:1.15; }
.hero-sub { margin:8px 0 16px; color:#D6E6F5; font-size:1rem; max-width:720px; }
.pill { display:inline-block; padding:4px 13px; border-radius:999px; background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.28); font-size:.76rem; margin:0 8px 6px 0; color:#fff; font-weight:500; }
.card { background:#fff; border:1px solid #E1E8F2; border-radius:16px; padding:16px; box-shadow:0 4px 18px rgba(16,42,67,.05); }
.card-title { font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:#5B6F88; font-weight:600; margin-bottom:10px; }
.imgwrap img { width:100%; border-radius:10px; display:block; background:#000; }
.caption { font-size:.78rem; color:#5B6F88; margin-top:8px; }
.legend { display:flex; align-items:center; gap:8px; font-size:.72rem; color:#5B6F88; margin-top:8px; }
.legend .grad { flex:1; height:8px; border-radius:6px; background:linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000); }
.result { display:flex; align-items:center; gap:28px; background:#fff; border:1px solid #E1E8F2; border-left:8px solid var(--c);
  border-radius:16px; padding:20px 28px; box-shadow:0 4px 18px rgba(16,42,67,.06); margin:6px 0 18px; }
.ring { width:106px; height:106px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.ring-in { width:82px; height:82px; border-radius:50%; background:#fff; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.ring-val { font-size:1.35rem; font-weight:700; color:#12263F; line-height:1; }
.ring-lbl { font-size:.62rem; color:#5B6F88; text-transform:uppercase; letter-spacing:.08em; margin-top:3px; }
.r-label { font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; color:#5B6F88; font-weight:600; }
.r-finding { font-size:1.9rem; font-weight:700; color:#12263F; letter-spacing:-.4px; line-height:1.2; }
.badge { display:inline-block; padding:4px 13px; border-radius:999px; font-size:.78rem; font-weight:600; color:#fff; background:var(--c); margin-right:8px; }
.r-note { color:#44566D; font-size:.9rem; margin-top:8px; }
.bar-row { margin:9px 0; }
.bar-top { display:flex; justify-content:space-between; font-size:.84rem; color:#12263F; font-weight:500; margin-bottom:4px; }
.bar-bg { height:9px; background:#E9EFF7; border-radius:6px; overflow:hidden; }
.bar-fg { height:100%; border-radius:6px; }
.q-row { display:flex; align-items:center; justify-content:space-between; padding:7px 0; border-bottom:1px solid #EEF2F8; font-size:.84rem; }
.q-row:last-child { border-bottom:none; }
.q-name { display:flex; align-items:center; gap:9px; color:#12263F; font-weight:500; }
.q-dot { width:9px; height:9px; border-radius:50%; }
.q-val { color:#5B6F88; }
.step { background:#fff; border:1px solid #E1E8F2; border-radius:16px; padding:20px; height:100%; box-shadow:0 4px 18px rgba(16,42,67,.05); }
.step-n { width:32px; height:32px; border-radius:10px; background:#E7F0FB; color:#0B5CAD; font-weight:700; display:flex; align-items:center; justify-content:center; margin-bottom:10px; }
.step-t { font-weight:600; color:#12263F; margin-bottom:4px; }
.step-d { font-size:.86rem; color:#5B6F88; line-height:1.5; }
.warn { background:#FFF7E6; border:1px solid #F4D79B; color:#7A5600; border-radius:12px; padding:11px 15px; font-size:.86rem; margin:6px 0 14px; }
.foot { text-align:center; color:#8A9BB1; font-size:.76rem; margin:30px 0 6px; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600; }
</style>
"""
st.markdown("\n".join(l.strip() for l in CSS.splitlines() if l.strip()), unsafe_allow_html=True)


def html(s: str):
    """Render HTML safely through markdown (no indentation / blank lines that markdown would break)."""
    st.markdown("\n".join(l.strip() for l in s.strip().splitlines() if l.strip()), unsafe_allow_html=True)


def data_uri(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- helpers
NORMAL_NAMES = {"normal", "healthy", "no finding", "no_finding", "negative"}
REVIEW_OPTIONS = ["Pending review", "Agree with AI finding", "Disagree with AI finding",
                  "Inconclusive - further imaging advised"]


def is_normal(label: str) -> bool:
    return label.strip().lower() in NORMAL_NAMES


def triage(label: str, conf: float, entropy: float, threshold: float):
    """Returns (title, colour, explanation). A triage *suggestion*, never a diagnosis."""
    if conf < threshold or entropy > 0.75:
        return "Manual review required", "#E8A317", "Model certainty is low. Interpret this study independently."
    if is_normal(label):
        return "Routine", "#1E9E6A", "No acute abnormality was flagged by the model."
    if conf >= 0.90:
        return "Priority review", "#D64545", "The model flagged a likely abnormality with high certainty."
    return "Expedited review", "#E8730C", "The model flagged a possible abnormality. Prompt review is advised."


def normalized_entropy(probs: dict) -> float:
    p = np.clip(np.array(list(probs.values())), 1e-9, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def load_image(file_bytes: bytes, name: str) -> Image.Image:
    if name.lower().endswith((".dcm", ".dicom")):
        import pydicom  # optional dependency
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] != 3:      # multi-frame (e.g. CT stack) -> middle slice
            arr = arr[arr.shape[0] // 2]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = 255 - arr
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(path, dev)
    return model, ckpt, dev


@st.cache_data(show_spinner=False, max_entries=20)
def run_analysis(file_bytes, file_name, target, alpha, ckpt_path, ckpt_mtime, _model, _ckpt, _dev):
    image = load_image(file_bytes, file_name)
    res = analyze(_model, _ckpt, image, _dev, target_class=target, alpha=alpha)
    big = np.array(image.resize((448, 448), Image.BICUBIC))
    cam_big = cv2.resize(res["heatmap"].astype(np.float32), (448, 448), interpolation=cv2.INTER_CUBIC)
    res["original_big"] = big
    res["overlay_big"] = overlay_cam(big, np.clip(cam_big, 0, 1), alpha)
    res["quality"] = assess_quality(image)
    res["entropy"] = normalized_entropy(res["probs"])
    return res


def img_card(title, arr, caption="", legend=False):
    leg = '<div class="legend"><span>Low influence</span><div class="grad"></div><span>High influence</span></div>' if legend else ""
    html(f'<div class="card"><div class="card-title">{title}</div><div class="imgwrap"><img src="{data_uri(arr)}"/></div>'
         f'<div class="caption">{caption}</div>{leg}</div>')


# ----------------------------------------------------------------------------- model
ckpt_default = str(config.CHECKPOINT)
if "history" not in st.session_state:
    st.session_state.history = {}

with st.sidebar:
    html('<div class="brand"><div class="brand-logo">🩻</div><div><div class="brand-name">MediScan AI</div>'
         '<div class="brand-sub">Clinical Imaging Assistant</div></div></div>')
    ckpt_path = st.text_input("Model checkpoint", ckpt_default)
    alpha = st.slider("Heatmap opacity", 0.10, 0.90, 0.45, 0.05)
    threshold = st.slider("Minimum confidence for automatic triage", 0.50, 0.95, 0.70, 0.05)

if not Path(ckpt_path).exists():
    html('<div class="hero"><div class="hero-title">MediScan AI</div><div class="hero-sub">No trained model found yet.</div></div>')
    st.error("A trained model is required before the dashboard can analyse studies.")
    st.markdown("**Next step:** train a model from the project folder, then refresh this page:\n\n"
                "```\npython -m src.train --model efficientnet_b0 --batch-size 16 --workers 0\n```")
    st.stop()

model, ckpt, dev = get_model(ckpt_path)
classes = ckpt["classes"]
val_recall = ckpt.get("val_recall")

with st.sidebar:
    html(f'<div class="side-box"><span class="dot"></span><b>Model online</b><br>'
         f'Architecture: {ckpt["model_name"]}<br>Classes: {", ".join(classes)}<br>'
         f'Compute: {"GPU" if dev.type == "cuda" else "CPU"}'
         + (f'<br>Validation sensitivity: {val_recall:.1%}' if val_recall is not None else "") + '</div>')
    html('<div class="side-box"><b>Privacy</b><br>Images are processed locally on this machine and are not '
         'uploaded or stored. Session data is cleared when the browser tab is closed.</div>')
    html(f'<div class="side-box"><b>Intended use</b><br>{config.DISCLAIMER}</div>')

# ----------------------------------------------------------------------------- header
html(f"""
<div class="hero">
  <div class="hero-title">MediScan AI</div>
  <div class="hero-sub">AI-assisted analysis of chest X-ray and CT studies with explainable heatmaps,
  image-quality checks, clinician review and exportable reports.</div>
  <span class="pill">Explainable AI (Grad-CAM)</span><span class="pill">Image quality checks</span>
  <span class="pill">PDF reporting</span><span class="pill">Research use only</span>
</div>
""")

tab_analysis, tab_history, tab_about = st.tabs(["Analysis", "Session log", "Model & performance"])

# ----------------------------------------------------------------------------- analysis tab
with tab_analysis:
    html('<div class="warn"><b>Decision support only.</b> This tool does not provide a diagnosis. '
         'All AI output must be reviewed and confirmed by a qualified clinician.</div>')

    with st.container(border=True):
        st.markdown("##### 1 · Study details")
        c1, c2, c3, c4 = st.columns([1.2, 0.7, 0.9, 1.2])
        patient_id = c1.text_input("Patient ID", placeholder="e.g. PT-10482")
        age = c2.number_input("Age", 0, 120, 45)
        sex = c3.selectbox("Sex", ["Not specified", "Female", "Male", "Other"])
        modality = c4.selectbox("Modality", ["Chest X-ray", "Chest CT (single slice)"])
        c5, c6 = st.columns([1, 2])
        clinician = c5.text_input("Requested by", placeholder="Referring clinician (optional)")
        history_txt = c6.text_area("Clinical history", placeholder="Symptoms, relevant history, indication for imaging...", height=80)

    with st.container(border=True):
        st.markdown("##### 2 · Upload study")
        file = st.file_uploader("Drag and drop a study, or browse", type=["jpg", "jpeg", "png", "bmp", "dcm"],
                                help="JPG, PNG, BMP or DICOM (.dcm). Maximum 50 MB.")

    if not file:
        st.write("")
        a, b, c = st.columns(3, gap="medium")
        for col, n, t, d in [(a, "1", "Enter study details", "Add patient information and clinical context so the report is complete."),
                             (b, "2", "Upload the image", "The model analyses the study and highlights the regions that drove its decision."),
                             (c, "3", "Review and export", "Confirm or challenge the AI finding, then download a PDF report.")]:
            with col:
                html(f'<div class="step"><div class="step-n">{n}</div><div class="step-t">{t}</div><div class="step-d">{d}</div></div>')
    else:
        file_bytes = file.getvalue()
        rid = "MS-" + hashlib.md5(file_bytes).hexdigest()[:8].upper()

        explain_for = st.radio("Explain heatmap for", ["Predicted class"] + classes, horizontal=True)
        target = None if explain_for == "Predicted class" else classes.index(explain_for)

        try:
            with st.spinner("Analysing study..."):
                r = run_analysis(file_bytes, file.name, target, alpha, ckpt_path,
                                 Path(ckpt_path).stat().st_mtime, model, ckpt, dev)
        except Exception as e:  # unreadable file, missing pydicom, etc.
            st.error(f"This file could not be analysed: {e}")
            st.stop()

        t_title, t_color, t_note = triage(r["pred_label"], r["confidence"], r["entropy"], threshold)
        pct = r["confidence"] * 100
        html(f"""
        <div class="result" style="--c:{t_color}">
          <div class="ring" style="background:conic-gradient({t_color} {pct:.1f}%, #E6ECF5 0)">
            <div class="ring-in"><div class="ring-val">{pct:.0f}%</div><div class="ring-lbl">confidence</div></div>
          </div>
          <div>
            <div class="r-label">AI finding</div>
            <div class="r-finding">{r["pred_label"]}</div>
            <div style="margin-top:8px"><span class="badge" style="--c:{t_color}">{t_title}</span>
            <span style="color:#5B6F88;font-size:.8rem">Report {rid}</span></div>
            <div class="r-note">{t_note}</div>
          </div>
        </div>
        """)

        col1, col2, col3 = st.columns([1, 1, 0.9], gap="medium")
        with col1:
            img_card("Original study", r["original_big"], f'{file.name}')
        with col2:
            img_card("Grad-CAM overlay", r["overlay_big"], f'Explaining: {r["cam_class"]}', legend=True)
        with col3:
            bars = ""
            for k, v in sorted(r["probs"].items(), key=lambda x: -x[1]):
                colr = t_color if k == r["pred_label"] else "#9DB4CF"
                bars += (f'<div class="bar-row"><div class="bar-top"><span>{k}</span><span>{v:.1%}</span></div>'
                         f'<div class="bar-bg"><div class="bar-fg" style="width:{v*100:.1f}%;background:{colr}"></div></div></div>')
            html(f'<div class="card"><div class="card-title">Class probabilities</div>{bars}</div>')
            st.write("")
            q = r["quality"]
            rows = "".join(
                f'<div class="q-row"><span class="q-name"><span class="q-dot" style="background:{"#1E9E6A" if c["ok"] else "#E8A317"}"></span>'
                f'{c["name"]}</span><span class="q-val" title="{"" if c["ok"] else c["hint"]}">{c["value"]}'
                f'{"" if c["ok"] else " ⚠"}</span></div>' for c in q["checks"])
            html(f'<div class="card"><div class="card-title">Image quality · {q["passed"]}/{q["total"]} passed</div>{rows}</div>')
        failed = [c for c in q["checks"] if not c["ok"]]
        if failed:
            html('<div class="warn"><b>Image quality notice:</b> ' + " ".join(c["hint"] for c in failed) + '</div>')

        st.write("")
        with st.container(border=True):
            st.markdown("##### 3 · Clinician review")
            rc1, rc2 = st.columns([1, 2])
            review_status = rc1.radio("Assessment", REVIEW_OPTIONS, key=f"rv_{rid}")
            review_notes = rc2.text_area("Comments", key=f"rn_{rid}", height=130,
                                         placeholder="Final impression, recommendations, follow-up...")
            b1, b2, b3 = st.columns([1, 1, 1])
            patient = {"id": patient_id.strip() or "Not provided", "age": age, "sex": sex,
                       "modality": modality, "clinician": clinician.strip(), "history": history_txt.strip()}
            pdf = build_pdf({"report_id": rid, "patient": patient, "finding": r["pred_label"],
                             "confidence": r["confidence"], "triage": t_title, "triage_note": t_note,
                             "probs": r["probs"], "quality": r["quality"], "original": r["original_big"],
                             "overlay": r["overlay_big"], "cam_class": r["cam_class"],
                             "review_status": review_status, "review_notes": review_notes.strip(),
                             "model_name": ckpt["model_name"], "classes": classes, "disclaimer": config.DISCLAIMER})
            b1.download_button("Download PDF report", pdf, f"MediScan_{rid}.pdf", "application/pdf", type="primary")
            ov = io.BytesIO(); Image.fromarray(r["overlay_big"]).save(ov, format="PNG")
            b2.download_button("Download heatmap (PNG)", ov.getvalue(), f"MediScan_{rid}_gradcam.png", "image/png")
            if b3.button("Save to session log"):
                st.session_state.history[rid] = {
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Report ID": rid,
                    "Patient ID": patient["id"], "Age": age, "Sex": sex, "Modality": modality,
                    "AI finding": r["pred_label"], "Confidence": f'{r["confidence"]:.1%}',
                    "Triage": t_title, "Clinician review": review_status}
                st.success("Saved to the session log.")

# ----------------------------------------------------------------------------- history tab
with tab_history:
    st.markdown("#### Session log")
    st.caption("Studies saved during this browser session. Nothing is written to disk or sent anywhere.")
    if st.session_state.history:
        df = pd.DataFrame(list(st.session_state.history.values()))
        m1, m2, m3 = st.columns(3)
        m1.metric("Studies saved", len(df))
        m2.metric("Flagged abnormal", int((~df["AI finding"].map(is_normal)).sum()))
        m3.metric("Reviewed by clinician", int((df["Clinician review"] != "Pending review").sum()))
        st.dataframe(df, hide_index=True)
        d1, d2 = st.columns([1, 5])
        d1.download_button("Export CSV", df.to_csv(index=False).encode(), "mediscan_session_log.csv", "text/csv")
        if d2.button("Clear log"):
            st.session_state.history = {}
            st.rerun()
    else:
        st.info("No studies saved yet. Analyse a study and press **Save to session log**.")

# ----------------------------------------------------------------------------- about tab
with tab_about:
    st.markdown("#### Model & performance")
    mfile = next((p for p in [config.OUTPUT_DIR / "metrics_test.json", config.OUTPUT_DIR / "metrics_val.json"] if p.exists()), None)
    a1, a2 = st.columns([1, 1], gap="large")
    with a1:
        st.markdown("**Model card**")
        st.markdown(f"- **Architecture:** {ckpt['model_name']} (ImageNet pre-trained, fine-tuned)\n"
                    f"- **Input size:** {ckpt['img_size']} × {ckpt['img_size']} px\n"
                    f"- **Classes:** {', '.join(classes)}\n"
                    f"- **Explainability:** Grad-CAM on the last convolutional block\n"
                    f"- **Training:** two-phase transfer learning, class-weighted loss, best checkpoint chosen by validation sensitivity")
    with a2:
        st.markdown("**Intended use and limitations**")
        st.markdown("- Research, education and workflow prototyping.\n"
                    "- Not validated on external hospital data; performance may drop on other scanners or populations.\n"
                    "- Heatmaps explain the model, not the patient; they are not lesion segmentations.\n"
                    "- Not a medical device. Clinical use would require prospective validation and regulatory approval.")
    st.divider()
    if mfile:
        mt = json.loads(mfile.read_text())
        st.markdown(f"**Measured performance** ({'held-out test set' if 'test' in mfile.name else 'validation set'})")
        k1, k2, k3 = st.columns(3)
        k1.metric("Accuracy", f'{mt["accuracy"]:.1%}')
        k2.metric("ROC-AUC (macro)", "n/a" if mt["roc_auc"] is None else f'{mt["roc_auc"]:.3f}')
        k3.metric("Classes", len(classes))
        pc = pd.DataFrame(mt["per_class"]).T.rename(columns={"sensitivity_recall": "Sensitivity",
                                                              "specificity": "Specificity", "precision": "Precision"})
        st.dataframe((pc * 100).round(1).astype(str) + " %")
        st.caption("Computed from your own training run. External validation on independent data is required before any clinical use.")
        imgs = [(config.OUTPUT_DIR / n, c) for n, c in [("confusion_test.png", "Confusion matrix"), ("roc_test.png", "ROC curves"),
                                                         ("training_curves.png", "Training curves")] if (config.OUTPUT_DIR / n).exists()]
        if imgs:
            for col, (p, cap) in zip(st.columns(len(imgs)), imgs):
                col.image(str(p), caption=cap)
    else:
        st.info("Performance metrics will appear here after training completes (`python -m src.train`).")

html(f'<div class="foot">{config.APP_NAME} v{config.APP_VERSION} · Research prototype · Not a medical device</div>')
