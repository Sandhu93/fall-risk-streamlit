"""
TUDO — Fall Risk Assessment (self-contained Streamlit deploy).

No FastAPI backend. Inference runs directly via FallRiskV3InferenceEngine.
Pages: Home · New Assessment · Results · Past Assessments · Live Camera · Next Steps
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import datetime
from pathlib import Path
from PIL import Image as _PILImage

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import av
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from inference import (
    DEFAULT_HIGH_THRESHOLD,
    DEFAULT_MED_THRESHOLD,
    WINDOW_SIZE,
    FallRiskV3InferenceEngine,
    FrameProcessorV3,
    GaitCNNv3Soft,
    _draw_overlay,
    detect_pose_yolo,
    draw_pose_landmarks,
    get_yolo_model,
    lm_to_dict,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "fall_risk_cnn_occu_v3_soft.pt"
ASSETS_PATH = Path(__file__).parent / "assets"

# ---------------------------------------------------------------------------
# Logo helpers
# ---------------------------------------------------------------------------
_LOGO_PATH = ASSETS_PATH / "logo.png"
_LOGO_B64: str | None = None
if _LOGO_PATH.exists():
    _LOGO_B64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()


def _logo_img(height: int, extra_style: str = "") -> str:
    """Return an <img> tag with the base64-encoded logo, or an emoji fallback."""
    if _LOGO_B64:
        return (
            f'<img src="data:image/png;base64,{_LOGO_B64}" '
            f'style="height:{height}px;width:auto;vertical-align:middle;{extra_style}">'
        )
    return "🌿"


# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------
_page_icon = _PILImage.open(_LOGO_PATH) if _LOGO_PATH.exists() else "🏥"

st.set_page_config(
    page_title="TUDO — Fall Risk Assessment",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------------------------------------------------------------
# Persistent history (JSON files next to this script)
# ---------------------------------------------------------------------------
_HISTORY_FILE = Path(__file__).parent / "assessments_history.json"
_LAST_ANALYSIS_FILE = Path(__file__).parent / "last_analysis.json"


def _load_history() -> list:
    try:
        if _HISTORY_FILE.exists():
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_history(records: list) -> None:
    try:
        _HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not save assessment history: {exc}")


def _load_last_analysis() -> dict | None:
    try:
        if _LAST_ANALYSIS_FILE.exists():
            data = json.loads(_LAST_ANALYSIS_FILE.read_text(encoding="utf-8"))
            return data if data else None
    except Exception:
        pass
    return None


def _save_last_analysis(analysis: dict) -> None:
    try:
        _LAST_ANALYSIS_FILE.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not save last analysis: {exc}")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "page": "home",
    "high_threshold": DEFAULT_HIGH_THRESHOLD,
    "med_threshold": DEFAULT_MED_THRESHOLD,
    "aggregation": "p90",
    "min_high_windows": 2,
    "show_pose": True,
    "past_assessments": _load_history(),
    "last_analysis": _load_last_analysis(),
    "last_annotated_bytes": None,
    "form_first_name": "",
    "form_last_name": "",
    "form_age": "",
    "form_sex": "",
    "form_date": str(datetime.date.today()),
    "form_diagnosis": "",
    "form_prev_falls": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---------------------------------------------------------------------------
# Global CSS — light-blue + sage-green clinical theme (TUDO)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide default Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0 !important; max-width: 100% !important; }

    /* ── Navigation bar ── */
    .tudo-nav {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 0.5rem;
        padding: 0.75rem 2rem;
        background: #ffffff;
        border-bottom: 1px solid #e0e6ed;
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .tudo-nav-logo {
        margin-right: auto;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        color: #2d7a4f;
        text-decoration: none;
    }
    .tudo-nav a {
        text-decoration: none;
        color: #444;
        font-size: 0.92rem;
        padding: 0.35rem 0.7rem;
        border-radius: 6px;
        transition: background 0.15s;
    }
    .tudo-nav a:hover { background: #f0f4f8; }
    .tudo-nav a.active { color: #2d7a4f; font-weight: 600; }
    .tudo-avatar {
        width: 34px; height: 34px;
        border-radius: 50%;
        background: #2d7a4f;
        color: white;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.9rem;
        margin-left: 0.5rem;
    }

    /* ── Hero banner ── */
    .tudo-hero {
        background: #78aec8;
        padding: 2.5rem 2rem 2rem;
        text-align: center;
        color: white;
    }
    .tudo-hero-logo {
        width: 72px; height: 72px;
        border-radius: 50%;
        background: white;
        margin: 0 auto 0.75rem;
        display: flex; align-items: center; justify-content: center;
        font-size: 2rem;
    }
    .tudo-hero h1 { font-size: 1.8rem; font-weight: 700; margin: 0; }

    /* ── Home action card ── */
    .home-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        max-width: 480px;
        margin: 2rem auto;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    .home-card h2 { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .home-card p { color: #555; font-size: 0.9rem; margin-bottom: 1.2rem; }
    .action-row {
        display: flex; align-items: center;
        border: 1px solid #e8ecf0;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
        cursor: pointer;
        transition: background 0.15s;
        text-decoration: none;
        color: inherit;
    }
    .action-row:hover { background: #f5f8fa; }
    .action-icon { font-size: 1.4rem; margin-right: 0.85rem; color: #78aec8; }
    .action-text strong { font-weight: 600; font-size: 0.95rem; }
    .action-text span { display: block; font-size: 0.82rem; color: #777; }
    .action-arrow { margin-left: auto; color: #aaa; }

    /* ── Assessment form ── */
    .form-wrapper {
        background: #a8bfa3;
        border-radius: 12px;
        padding: 1.75rem 2rem 1.5rem;
        max-width: 460px;
        margin: 1.5rem auto;
    }

    /* ── Results metrics bar ── */
    .metrics-bar {
        display: flex;
        gap: 2rem;
        align-items: baseline;
        padding: 0.75rem 1.5rem;
        border-bottom: 1px solid #e5e9ed;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .metrics-bar .risk-low  { color: #2ecc71; font-size: 1.4rem; }
    .metrics-bar .risk-med  { color: #f39c12; font-size: 1.4rem; }
    .metrics-bar .risk-high { color: #e74c3c; font-size: 1.4rem; }
    .metrics-bar .mval { font-size: 1.15rem; color: #222; }
    .metrics-bar .mlabel { font-size: 0.75rem; color: #888; display: block; }

    /* ── Next steps ── */
    .next-steps { padding: 1rem 2rem; }
    .ns-low  { color: #2d7a4f; font-weight: 600; }
    .ns-med  { color: #c47e00; font-weight: 600; }
    .ns-high { color: #c0392b; font-weight: 600; }

    /* ── Past assessment row ── */
    .past-row {
        border: 1px solid #dde3ea;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        background: white;
    }
    .risk-badge {
        border-radius: 5px;
        padding: 0.2rem 0.55rem;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
    }
    .badge-LOW    { background: #2ecc71; }
    .badge-MEDIUM { background: #f39c12; }
    .badge-HIGH   { background: #e74c3c; }
    .badge-UNKNOWN { background: #95a5a6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_engine() -> FallRiskV3InferenceEngine:
    """Load the inference engine once and cache for all sessions."""
    return FallRiskV3InferenceEngine(model_path=MODEL_PATH)


@st.cache_resource
def _load_model_components():
    """Load raw model weights for webcam processor (cached separately)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = GaitCNNv3Soft().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm_mu = np.array(ckpt["norm_mu"], dtype=np.float32)
    norm_sd = np.array(ckpt["norm_sd"], dtype=np.float32)
    return model, device, norm_mu, norm_sd


# ---------------------------------------------------------------------------
# Webcam video processor (streamlit-webrtc)
# ---------------------------------------------------------------------------


class FallRiskWebcamProcessor(VideoProcessorBase):
    def __init__(self):
        model, device, norm_mu, norm_sd = _load_model_components()
        self.fp = FrameProcessorV3(
            model=model,
            device=device,
            norm_mu=norm_mu,
            norm_sd=norm_sd,
            high_threshold=st.session_state.get("high_threshold", DEFAULT_HIGH_THRESHOLD),
            med_threshold=st.session_state.get("med_threshold", DEFAULT_MED_THRESHOLD),
            aggregation=st.session_state.get("aggregation", "p90"),
            min_high_windows=st.session_state.get("min_high_windows", 2),
        )
        self.show_pose = st.session_state.get("show_pose", True)
        self._yolo = get_yolo_model()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        kp = detect_pose_yolo(self._yolo, img)
        lm_dict = lm_to_dict(kp, w, h) if kp is not None else None
        if self.show_pose and kp is not None:
            draw_pose_landmarks(img, kp)
        self.fp.update(lm_dict)
        img = _draw_overlay(
            img, self.fp.current_risk, self.fp.current_probs,
            self.fp.high_threshold, self.fp.frame_count,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------------------------------------------------------------------
# Direct inference helper (replaces HTTP calls to FastAPI)
# ---------------------------------------------------------------------------


def run_inference(
    video_bytes: bytes,
    filename: str,
    high_threshold: float,
    med_threshold: float,
    aggregation: str,
    min_high_windows: int,
    show_pose: bool,
) -> tuple[bytes, dict]:
    """Write video to a temp file, run annotated inference, return (annotated_bytes, analysis)."""
    engine = _load_engine()
    # Update engine parameters for this call
    engine.high_threshold = high_threshold
    engine.med_threshold = med_threshold
    engine.aggregation = aggregation
    engine.min_high_windows = min_high_windows

    suffix = Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_in:
        f_in.write(video_bytes)
        in_path = f_in.name

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_out:
        out_path = f_out.name

    analysis = engine.process_video_annotated(in_path, out_path, show_pose=show_pose)
    annotated_bytes = Path(out_path).read_bytes()
    return annotated_bytes, analysis


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def risk_color(risk: str) -> str:
    return {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}.get(risk, "#95a5a6")


def risk_css_class(risk: str) -> str:
    return {"HIGH": "risk-high", "MEDIUM": "risk-med", "LOW": "risk-low"}.get(risk, "")


def nav_bar(active: str) -> None:
    pages = [
        ("home", "Home"),
        ("past", "Past Assessment"),
        ("new", "New Assessment"),
        ("results", "Results"),
        ("webcam", "Live Camera"),
    ]
    links = ""
    for key, label in pages:
        cls = "active" if key == active else ""
        links += f'<a class="{cls}" href="?page={key}" onclick="void(0)">{label}</a>'
    st.markdown(
        f"""
        <div class="tudo-nav">
            <span class="tudo-nav-logo">{_logo_img(28)} TUDO</span>
            {links}
            <div class="tudo-avatar">A</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plotly_timeline(window_results: list, high_th: float, med_th: float) -> None:
    if not window_results:
        st.info("No windows detected.")
        return
    df = pd.DataFrame(window_results)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["frame_idx"], y=df["prob_high"],
        name="prob_high", line=dict(color="#e74c3c", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["frame_idx"], y=df["prob_medium"],
        name="prob_medium", line=dict(color="#f39c12", width=2),
    ))
    fig.add_hline(
        y=high_th, line_dash="dash", line_color="#999",
        annotation_text=f"high_threshold={high_th:.2f}",
        annotation_position="top right", annotation_font_size=11,
    )
    fig.add_hline(
        y=med_th, line_dash="dot", line_color="#bbb",
        annotation_text=f"med_threshold={med_th:.2f}",
        annotation_position="bottom right", annotation_font_size=11,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=280,
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(range=[0, max(df["prob_high"].max(), high_th) * 1.15 + 0.05]),
        xaxis_title="Frame",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    high_n = sum(1 for w in window_results if w["risk"] == "HIGH")
    med_n = sum(1 for w in window_results if w["risk"] == "MEDIUM")
    st.caption(
        f"HIGH windows: **{high_n}** · MEDIUM windows: **{med_n}** · "
        f"Total: **{len(window_results)}** · "
        f"Peak P(HIGH): **{max(w['prob_high'] for w in window_results):.3f}**"
    )


# ===========================================================================
# Page routing via URL query param  (?page=...)
# ===========================================================================
query_params = st.query_params
_qpage = query_params.get("page", st.session_state["page"])
if isinstance(_qpage, list):
    _qpage = _qpage[0]
st.session_state["page"] = _qpage
page = st.session_state["page"]

# ---------------------------------------------------------------------------
# Sidebar — model parameters (no backend URL)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Advanced Settings")
    st.divider()
    st.session_state["high_threshold"] = st.slider(
        "HIGH Threshold", 0.05, 0.95,
        float(st.session_state["high_threshold"]), 0.01,
        help=f"Default: {DEFAULT_HIGH_THRESHOLD}. Lower = more sensitive to HIGH risk.",
    )
    st.session_state["med_threshold"] = st.slider(
        "MEDIUM Threshold", 0.05, 0.95,
        float(st.session_state["med_threshold"]), 0.01,
        help=f"Default: {DEFAULT_MED_THRESHOLD}.",
    )
    st.session_state["aggregation"] = st.selectbox(
        "Aggregation", ["p90", "p75", "max", "mean"],
        index=["p90", "p75", "max", "mean"].index(st.session_state["aggregation"]),
        help="How recent window P(HIGH) scores are combined.",
    )
    st.session_state["min_high_windows"] = st.number_input(
        "Min HIGH Windows", 1, 10, int(st.session_state["min_high_windows"]),
    )
    st.session_state["show_pose"] = st.toggle(
        "Show Pose Skeleton", value=bool(st.session_state["show_pose"]),
    )

high_threshold = st.session_state["high_threshold"]
med_threshold = st.session_state["med_threshold"]
aggregation = st.session_state["aggregation"]
min_high_windows = st.session_state["min_high_windows"]
show_pose = st.session_state["show_pose"]


# ===========================================================================
# HOME PAGE
# ===========================================================================
if page == "home":
    nav_bar("home")

    st.markdown(
        f"""
        <div class="tudo-hero">
            <div class="tudo-hero-logo">{_logo_img(56)}</div>
            <h1>TUDO</h1>
            <p style="margin:0.3rem 0 0;font-size:1rem;opacity:0.9;font-weight:400;">Fall Risk Assessment</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-card">
            <h2>Gait Fall Risk Tool</h2>
            <p>TUDO is an AI-powered screening application that analyzes gait patterns from uploaded
            walking videos to estimate fall risk. The system uses computer vision and machine learning
            to evaluate movement, balance, and walking stability, generating a simple and interpretable
            fall risk score (Low, Medium, High). The tool supports early detection and preventive care
            by enabling accessible at-home gait assessment.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        if st.button(
            "📝  Start New Assessment\nEnter patient information, gait parameters, and upload a gait video.",
            use_container_width=True,
        ):
            st.query_params["page"] = "new"
            st.rerun()

        if st.button(
            "📋  View Past Assessments\nBrowse previously recorded gait assessments and fall risk categories.",
            use_container_width=True,
        ):
            st.query_params["page"] = "past"
            st.rerun()


# ===========================================================================
# NEW ASSESSMENT PAGE
# ===========================================================================
elif page == "new":
    nav_bar("new")

    st.markdown("### New Gait Assessment")
    st.caption("Please complete the assessment questions below.")

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.form("assessment_form"):
            st.markdown('<div class="form-wrapper">', unsafe_allow_html=True)

            col_fn, col_ln = st.columns(2)
            with col_fn:
                first_name = st.text_input(
                    "First Name", placeholder="First Name",
                    value=st.session_state["form_first_name"],
                )
            with col_ln:
                last_name = st.text_input(
                    "Last Name", placeholder="Last Name",
                    value=st.session_state["form_last_name"],
                )

            age = st.text_input(
                "Age (Years)", placeholder="e.g. 67",
                value=st.session_state["form_age"],
            )
            sex = st.text_input(
                "Sex / Gender", placeholder="e.g. Female, Male, Intersex, Other",
                value=st.session_state["form_sex"],
            )
            assess_date = st.date_input("Assessment Date", value=datetime.date.today())
            diagnosis = st.text_input(
                "Diagnosis", placeholder="Optional clinical summary",
                value=st.session_state["form_diagnosis"],
            )
            prev_falls = st.text_input(
                "Previous Falls (Last 12 Months)", placeholder="e.g. 2",
                value=st.session_state["form_prev_falls"],
            )
            uploaded = st.file_uploader(
                "Video Upload",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="10–60 s clips work best. Subject should be fully visible.",
            )

            st.markdown('</div>', unsafe_allow_html=True)

            col_save, col_submit = st.columns(2)
            with col_save:
                save_btn = st.form_submit_button("Save", use_container_width=True)
            with col_submit:
                submit_btn = st.form_submit_button("Submit", type="primary", use_container_width=True)

        if save_btn or submit_btn:
            st.session_state.update({
                "form_first_name": first_name,
                "form_last_name": last_name,
                "form_age": age,
                "form_sex": sex,
                "form_date": str(assess_date),
                "form_diagnosis": diagnosis,
                "form_prev_falls": prev_falls,
            })

        if submit_btn:
            if not uploaded:
                st.warning("Please upload a gait video to run the analysis.")
            else:
                with st.spinner("Analysing gait video — this may take a minute…"):
                    annotated_bytes, analysis = run_inference(
                        video_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        high_threshold=high_threshold,
                        med_threshold=med_threshold,
                        aggregation=aggregation,
                        min_high_windows=min_high_windows,
                        show_pose=show_pose,
                    )
                analysis["patient_name"] = f"{first_name} {last_name}".strip()
                analysis["assess_date"] = str(assess_date)
                st.session_state["last_analysis"] = analysis
                st.session_state["last_annotated_bytes"] = annotated_bytes
                _save_last_analysis(analysis)

                new_record = {
                    "name": f"{first_name} {last_name}".strip() or "Unknown",
                    "date": str(assess_date),
                    "risk": analysis.get("final_risk", "UNKNOWN"),
                    "age": age,
                    "sex": sex,
                    "diagnosis": diagnosis,
                    "prev_falls": prev_falls,
                }
                st.session_state["past_assessments"].insert(0, new_record)
                _save_history(st.session_state["past_assessments"])
                st.query_params["page"] = "results"
                st.rerun()


# ===========================================================================
# RESULTS PAGE
# ===========================================================================
elif page == "results":
    nav_bar("results")

    analysis = st.session_state.get("last_analysis")
    if analysis is None:
        analysis = _load_last_analysis()
        if analysis is not None:
            st.session_state["last_analysis"] = analysis
    annotated_bytes = st.session_state.get("last_annotated_bytes")

    if analysis is None:
        st.info("No results yet. Submit a New Assessment first.")
        if st.button("Go to New Assessment"):
            st.query_params["page"] = "new"
            st.rerun()
    else:
        risk = analysis.get("final_risk", "INSUFFICIENT_DATA")
        agg_high = analysis.get("aggregated_high_score")
        window_results = analysis.get("window_results", [])
        ht = analysis.get("high_threshold", high_threshold)
        mt = analysis.get("med_threshold", med_threshold)
        windows = analysis.get("windows_processed", 0)
        det_rate = analysis.get("detection_rate", 0)
        proc_time = analysis.get("processing_time_s", 0)

        risk_cls = risk_css_class(risk)
        st.markdown(
            f"""
            <div class="metrics-bar">
                <span class="{risk_cls}">{risk}</span>
                <span><span class="mlabel">P(HIGH) Agg.</span>
                      <span class="mval">{f"{agg_high:.3f}" if agg_high is not None else "N/A"}</span></span>
                <span><span class="mlabel">HIGH Thr.</span>
                      <span class="mval">{ht:.2f}</span></span>
                <span><span class="mlabel">Windows</span>
                      <span class="mval">{windows}</span></span>
                <span><span class="mlabel">Pose Det.</span>
                      <span class="mval">{det_rate:.1%}</span></span>
                <span><span class="mlabel">Time</span>
                      <span class="mval">{proc_time:.1f} s</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("")

        vid_col, chart_col = st.columns([3, 2])

        with vid_col:
            st.markdown("**Annotated Video** (H.264, plays in browser)")
            if annotated_bytes:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(annotated_bytes)
                    tmp_path = f.name
                st.video(tmp_path)
                st.download_button(
                    "⬇ Download Annotated MP4",
                    data=annotated_bytes,
                    file_name="annotated_gait.mp4",
                    mime="video/mp4",
                )

        with chart_col:
            st.markdown("**Risk Probability Timeline**")
            plotly_timeline(window_results, ht, mt)
            if window_results:
                with st.expander("Per-window table"):
                    df = pd.DataFrame(window_results)[
                        ["frame_idx", "prob_low", "prob_medium", "prob_high", "risk"]
                    ]
                    df.columns = ["Frame", "P(LOW)", "P(MED)", "P(HIGH)", "Risk"]

                    def _hl(r):
                        bg = {"HIGH": "#ffcccc", "MEDIUM": "#ffe8cc", "LOW": "#ccffcc"}.get(r["Risk"], "")
                        return [f"background-color:{bg}"] * len(r)

                    st.dataframe(
                        df.style.apply(_hl, axis=1),
                        use_container_width=True, hide_index=True,
                    )

        with st.expander("Raw analysis JSON"):
            st.json(analysis)

        st.divider()

        if st.button("View Next Steps →", use_container_width=False):
            st.query_params["page"] = "nextsteps"
            st.rerun()


# ===========================================================================
# PAST ASSESSMENTS PAGE
# ===========================================================================
elif page == "past":
    nav_bar("past")

    st.markdown("### Past Assessments")
    past = st.session_state.get("past_assessments", [])

    if not past:
        st.info("No past assessments yet. Submit a New Assessment to see history here.")
        if st.button("Start New Assessment"):
            st.query_params["page"] = "new"
            st.rerun()
    else:
        for i, rec in enumerate(past):
            risk = rec.get("risk", "UNKNOWN")
            badge_cls = f"badge-{risk}" if risk in ("LOW", "MEDIUM", "HIGH") else "badge-UNKNOWN"
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(
                    f"""
                    <div class="past-row">
                        <span class="risk-badge {badge_cls}">{risk}</span>
                        <span><strong>{rec.get('name', 'Unknown')}</strong>
                              &nbsp;·&nbsp; Age: {rec.get('age', '—')}
                              &nbsp;·&nbsp; {rec.get('sex', '—')}</span>
                        <span style="margin-left:auto;color:#888;font-size:0.85rem">{rec.get('date', '')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ===========================================================================
# NEXT STEPS PAGE
# ===========================================================================
elif page == "nextsteps":
    nav_bar("results")

    _grid_images = [
        ASSETS_PATH / "1.jpg",
        ASSETS_PATH / "2.jpg",
        ASSETS_PATH / "3.jpg",
        ASSETS_PATH / "4.jpg",
        ASSETS_PATH / "5.jpg",
        ASSETS_PATH / "Untitled design (1).jpg",
    ]

    def _square_b64(path: Path, size: int = 160) -> str | None:
        """Return base64 PNG of a centre-cropped square thumbnail."""
        if not path.exists():
            return None
        img = _PILImage.open(path).convert("RGB")
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((size, size), _PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    _THUMB_SIZE = 160
    _cells_html = ""
    for img_path in _grid_images:
        b64 = _square_b64(img_path, _THUMB_SIZE)
        if b64:
            _cells_html += (
                f'<div style="width:{_THUMB_SIZE}px;height:{_THUMB_SIZE}px;">'
                f'<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;height:100%;object-fit:cover;border-radius:8px;">'
                f'</div>'
            )

    if _cells_html:
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(3,{_THUMB_SIZE}px);'
            f'gap:10px;width:fit-content;">{_cells_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    _ns_analysis = st.session_state.get("last_analysis") or _load_last_analysis()
    _ns_risk = (_ns_analysis or {}).get("final_risk", "")

    def _ns_highlight(level: str) -> str:
        if _ns_risk == level:
            mapping = {"LOW": "#d4f8e8", "MEDIUM": "#fff3cd", "HIGH": "#fde8e8"}
            return f"background:{mapping[level]};border-radius:6px;padding:0.4rem 0.6rem;display:block;"
        return ""

    _style_low = _ns_highlight("LOW")
    _style_med = _ns_highlight("MEDIUM")
    _style_high = _ns_highlight("HIGH")

    st.markdown("### Next Steps?")
    st.caption("These are the recommended next steps to follow based off of your risk score.")

    st.markdown(
        f'<span style="{_style_low}">'
        '<span class="ns-low">Low Risk:</span> Your walking pattern appears stable, and your likelihood of '
        "falling is minimal. Continue your regular activities and maintain a healthy lifestyle.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        f'<span style="{_style_med}">'
        '<span class="ns-med">Medium Risk:</span> Some aspects of your gait may indicate a moderate risk of '
        "falls. It is recommended to practice balance and strengthening exercises, review your home for "
        "potential hazards, and consult a healthcare professional if necessary.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        f'<span style="{_style_high}">'
        '<span class="ns-high">High Risk:</span> Your walking pattern suggests a higher chance of falls. '
        "You should consider consulting a healthcare provider promptly, using assistive devices if needed, "
        "implementing safety measures at home, and engaging in targeted balance and strength exercises.</span>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    if st.button("← Back to Results"):
        st.query_params["page"] = "results"
        st.rerun()


# ===========================================================================
# LIVE WEBCAM PAGE
# ===========================================================================
elif page == "webcam":
    nav_bar("webcam")
    st.subheader("Real-time Fall Risk — Live Webcam")
    st.caption(
        "Pose estimation and inference run locally in the browser tab. "
        f"Model needs **{WINDOW_SIZE} frames** (~3 s at 30 fps) before first prediction."
    )
    st.info(
        "**Note:** Live webcam requires WebRTC support. "
        "It works in local deployments; on Streamlit Cloud it may require a TURN server "
        "or may be limited by the hosting environment."
    )

    ctx = webrtc_streamer(
        key="fall-risk-v3-webcam",
        video_processor_factory=FallRiskWebcamProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing and ctx.video_processor:
        st.divider()
        proc: FallRiskWebcamProcessor = ctx.video_processor
        stats_placeholder = st.empty()

        with stats_placeholder.container():
            risk = proc.fp.current_risk
            probs = proc.fp.current_probs
            frames = proc.fp.frame_count

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Risk", risk)
            c2.metric("P(HIGH)", f"{probs[2]:.3f}" if probs is not None else "--")
            c3.metric("P(MED)", f"{probs[1]:.3f}" if probs is not None else "--")
            c4.metric("Frames Processed", frames)

            color = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#2ecc71"}.get(risk, "#95a5a6")
            st.markdown(
                f"<div style='background:{color};padding:14px;border-radius:8px;"
                f"text-align:center;color:white;font-size:1.3rem;font-weight:bold'>"
                f"{risk}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("Click **START** above to begin live webcam analysis.")
