"""
Streamlit Evaluation App — Dual-SR vs Single-SR vs Bicubic

Usage:
    cd src
    streamlit run app.py

Workflow:
    1. Upload a high-resolution image (used as HR ground truth).
    2. App synthesises LR1 (bicubic ×4) and LR2 (Gaussian σ + bicubic ×4).
    3. Bicubic baseline: bicubic-upscale of LR1.
    4. Single-EDSR: feeds LR1 only.
    5. Dual-EDSR:   feeds LR1 + LR2.
    6. Shows all outputs side-by-side with PSNR / SSIM vs HR.
"""

import sys
import io
import hashlib
from pathlib import Path

import numpy as np
import cv2
import torch
import yaml
from PIL import Image
import streamlit as st

# ── make local packages importable when run from src/ ──────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from models.dual_edsr import DualEDSR, SingleEDSR
from utils.metrics import calculate_psnr, calculate_ssim, calculate_niqe, calculate_brisque

# ── constants ───────────────────────────────────────────────────────────────
CONFIG_PATH         = _HERE / "config" / "config.yaml"
DEFAULT_DUAL_CKPT   = _HERE / "checkpoints" / "dual"   / "dual_edsr_best.pth"
DEFAULT_SINGLE_CKPT = _HERE / "checkpoints" / "single" / "single_edsr_best.pth"


# ── helpers ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading Dual-EDSR model…")
def load_dual_model(ckpt_path: str, config: dict, device_str: str) -> DualEDSR:
    cfg = config["model"]
    model = DualEDSR(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        num_features=cfg.get("num_features", 64),
        num_residual_blocks=cfg.get("num_residual_blocks", 16),
        scale=cfg.get("scale", 4),
        res_scale=cfg.get("res_scale", 0.1),
        use_mean_shift=cfg.get("use_mean_shift", True),
    )
    ckpt  = torch.load(ckpt_path, map_location=device_str, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device_str).eval()
    return model


@st.cache_resource(show_spinner="Loading Single-EDSR model…")
def load_single_model(ckpt_path: str, config: dict, device_str: str) -> SingleEDSR:
    cfg = config["model"]
    model = SingleEDSR(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 3),
        num_features=cfg.get("num_features", 64),
        num_residual_blocks=cfg.get("num_residual_blocks", 16),
        scale=cfg.get("scale", 4),
        res_scale=cfg.get("res_scale", 0.1),
        use_mean_shift=cfg.get("use_mean_shift", True),
    )
    ckpt  = torch.load(ckpt_path, map_location=device_str, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device_str).eval()
    return model


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → (1, 3, H, W) float32 in [0, 1]."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1,3,H,W) or (3,H,W) float32 → PIL RGB."""
    arr = (t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def synthesise_lr_pair(
    hr: Image.Image, scale: int = 4, blur_sigma: float = 1.5
) -> tuple:
    hr_np       = np.array(hr.convert("RGB"))
    h, w        = hr_np.shape[:2]
    lr_w, lr_h  = w // scale, h // scale
    lr1         = Image.fromarray(hr_np).resize((lr_w, lr_h), Image.BICUBIC)
    blurred     = cv2.GaussianBlur(hr_np, (0, 0), float(blur_sigma))
    lr2         = Image.fromarray(blurred).resize((lr_w, lr_h), Image.BICUBIC)
    return lr1, lr2


def bicubic_up(lr: Image.Image, scale: int = 4) -> Image.Image:
    w, h = lr.size
    return lr.resize((w * scale, h * scale), Image.BICUBIC)


def run_dual(model: DualEDSR, lr1: Image.Image, lr2: Image.Image, device: str) -> Image.Image:
    with torch.no_grad():
        out = model(pil_to_tensor(lr1).to(device), pil_to_tensor(lr2).to(device))
    return tensor_to_pil(out)


def run_single(model: SingleEDSR, lr1: Image.Image, device: str) -> Image.Image:
    with torch.no_grad():
        out = model(pil_to_tensor(lr1).to(device))
    return tensor_to_pil(out)


def compute_metrics(sr: Image.Image, hr: Image.Image) -> tuple:
    """Returns (PSNR dB, SSIM) for SR vs HR."""
    hr_t = pil_to_tensor(hr)
    sr_t = pil_to_tensor(sr)
    return float(calculate_psnr(sr_t, hr_t)), float(calculate_ssim(sr_t, hr_t))


def metric_html(label: str, psnr: float, ssim: float, delta_p: float | None = None) -> None:
    delta_html = ""
    if delta_p is not None:
        color = "#a6e3a1" if delta_p >= 0 else "#f38ba8"
        sign = "+" if delta_p >= 0 else ""
        delta_html = (
            f'<span class="sr-delta" style="color:{color}">'
            f'({sign}{delta_p:.2f} dB vs bicubic)'
            f"</span>"
        )

    st.markdown(
        f"""
        <div class="sr-card">
          <div class="sr-card-title">{label} {delta_html}</div>
          <div class="sr-row">
            <div class="sr-kpi"><span class="sr-kpi-label">PSNR</span><b>{psnr:.2f} dB</b></div>
            <div class="sr-kpi"><span class="sr-kpi-label">SSIM</span><b>{ssim:.4f}</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pixel_fingerprint(img: Image.Image) -> str:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h = hashlib.sha1(arr.tobytes()).hexdigest()[:12]
    return f"{img.width}×{img.height} | {h}"


# ── page setup ───────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        """
        <style>
          [data-testid="block-container"] {
            max-width: 1200px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
          }
          [data-testid="stSidebarContent"] {
            padding-top: 1.5rem;
          }
          [data-testid="stButton"] button {
            border-radius: 10px;
            padding: 0.65rem 1rem;
          }

          /* Button theming (override Streamlit theme accent) */
          :root {
            /* Neutral primary: visually matches surrounding controls */
            --sr-primary-bg: rgba(255, 255, 255, 0.04);
            --sr-primary-bg-hover: rgba(255, 255, 255, 0.07);
            --sr-primary-border: rgba(120, 120, 120, 0.30);
            --sr-primary-border-hover: rgba(170, 170, 170, 0.45);

            --sr-secondary-bg: rgba(255, 255, 255, 0.04);
            --sr-secondary-bg-hover: rgba(255, 255, 255, 0.06);
            --sr-secondary-border: rgba(120, 120, 120, 0.30);
          }

          /* Primary buttons (Streamlit DOM varies across versions, so match multiple selectors) */
          div[data-testid="stBaseButton-primary"] button,
          div[data-testid="baseButton-primary"] button,
          .stButton > button[kind="primary"],
          button[kind="primary"] {
            background: var(--sr-primary-bg) !important;
            border: 1px solid var(--sr-primary-border) !important;
            color: rgba(255, 255, 255, 0.92) !important;
            box-shadow: none !important;
            background-image: none !important;
          }
          div[data-testid="stBaseButton-primary"] button:hover,
          div[data-testid="baseButton-primary"] button:hover,
          .stButton > button[kind="primary"]:hover,
          button[kind="primary"]:hover {
            background: var(--sr-primary-bg-hover) !important;
            border-color: var(--sr-primary-border-hover) !important;
          }

          /* Secondary buttons */
          div[data-testid="stBaseButton-secondary"] button,
          div[data-testid="baseButton-secondary"] button,
          .stButton > button[kind="secondary"],
          button[kind="secondary"] {
            background: var(--sr-secondary-bg) !important;
            border: 1px solid var(--sr-secondary-border) !important;
            color: rgba(255, 255, 255, 0.90) !important;
            box-shadow: none !important;
            background-image: none !important;
          }
          div[data-testid="stBaseButton-secondary"] button:hover,
          div[data-testid="baseButton-secondary"] button:hover,
          .stButton > button[kind="secondary"]:hover,
          button[kind="secondary"]:hover {
            background: var(--sr-secondary-bg-hover) !important;
          }

          div[data-testid="stBaseButton-primary"] button:disabled,
          div[data-testid="baseButton-primary"] button:disabled,
          div[data-testid="stBaseButton-secondary"] button:disabled,
          div[data-testid="baseButton-secondary"] button:disabled,
          button[kind="primary"]:disabled,
          button[kind="secondary"]:disabled {
            opacity: 0.55;
          }

          img {
            border-radius: 12px;
          }

          .sr-hero {
            border-radius: 16px;
            padding: 14px 18px;
            margin-bottom: 1.25rem;
            border: 1px solid rgba(120, 120, 120, 0.25);
            background: rgba(255, 255, 255, 0.03);
          }
          .sr-hero-title {
            font-size: 2.1rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            line-height: 1.15;
          }
          .sr-hero-subtitle {
            margin-top: 6px;
            opacity: 0.80;
            font-size: 0.98rem;
          }

          .sr-pill {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(120, 120, 120, 0.25);
            background: rgba(255, 255, 255, 0.03);
            font-size: 0.92rem;
          }

          .sr-card {
            border-radius: 12px;
            padding: 12px 16px;
            margin-top: 6px;
            border: 1px solid rgba(120, 120, 120, 0.25);
            background: rgba(255, 255, 255, 0.03);
          }
          .sr-card-title {
            font-weight: 650;
            margin-bottom: 6px;
          }
          .sr-row {
            display: flex;
            gap: 18px;
            flex-wrap: wrap;
          }
          .sr-kpi {
            font-variant-numeric: tabular-nums;
          }
          .sr-kpi-label {
            opacity: 0.7;
            margin-right: 8px;
          }
          .sr-delta {
            font-size: 0.85em;
            margin-left: 8px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Super-Resolution Evaluation",
    page_icon="SR",
    layout="wide",
)

inject_css()

st.markdown(
    """
    <div class="sr-hero">
      <div class="sr-hero-title">Dual-Image Super-Resolution</div>
      <div class="sr-hero-subtitle">
        Full-reference mode reports PSNR/SSIM against a provided ground truth. Blind mode reports NIQE/BRISQUE (lower is better).
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

config      = load_config()
scale       = config["model"].get("scale", 4)
blur_sigma  = config["data"].get("blur_sigma", 1.5)
device_str  = "cuda" if torch.cuda.is_available() else "cpu"

# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    st.subheader("Checkpoints")
    dual_ckpt_input   = st.text_input("Dual-EDSR checkpoint",   value=str(DEFAULT_DUAL_CKPT))
    single_ckpt_input = st.text_input("Single-EDSR checkpoint", value=str(DEFAULT_SINGLE_CKPT))

    st.subheader("Degradation")
    blur_sigma = st.slider(
        "LR2 Gaussian σ",
        0.5,
        5.0,
        float(blur_sigma),
        0.1,
        help="Must match the sigma used during training (default 1.5)",
    )

    st.subheader("Device")
    st.markdown(
        f"<div class='sr-pill'>Device: <b>{device_str.upper()}</b></div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption(
        "Checkpoint files default to `src/checkpoints/`.  \n"
        "Run `train.py` and `train_single.py` first to generate them."
    )

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_eval, tab_blind = st.tabs(["Full-Reference", "Blind (Sentinel-2)"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Full-Reference Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown(
        "Upload a **high-resolution image** as ground truth. "
        "The app synthesises LR1 (bicubic ÷4) and LR2 (Gaussian + bicubic ÷4) "
        "and compares Bicubic / Single-EDSR / Dual-EDSR."
    )

    # ── file upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload HR image (PNG / JPG / TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="hr_upload",
    )

    if uploaded is None:
        st.info("Upload a high-resolution image to start.")
    else:
        hr_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
        w, h   = hr_pil.size
        w_crop = (w // scale) * scale
        h_crop = (h // scale) * scale
        hr_pil = hr_pil.crop((0, 0, w_crop, h_crop))

        st.success(
            f"Loaded {uploaded.name} — {w_crop}×{h_crop} px → "
            f"LR size: {w_crop//scale}×{h_crop//scale} px (×{scale})"
        )

        lr1_pil, lr2_pil = synthesise_lr_pair(hr_pil, scale=scale, blur_sigma=blur_sigma)

        st.subheader("Inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(hr_pil,  caption="HR — ground truth", use_container_width=True)
        with c2:
            st.image(lr1_pil, caption=f"LR1 — bicubic ÷{scale}", use_container_width=True)
        with c3:
            st.image(lr2_pil, caption=f"LR2 — Gaussian(σ={blur_sigma:.1f}) + bicubic ÷{scale}",
                     use_container_width=True)

        with st.expander("Exports / diagnostics", expanded=False):
            st.caption("Use these exported LR files to reproduce the exact same inputs in the Blind tab.")
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                st.download_button("Download HR (cropped)", pil_to_bytes(hr_pil), "input_hr.png", "image/png")
                st.caption(f"HR: {pixel_fingerprint(hr_pil)}")
            with ex2:
                st.download_button("Download LR1", pil_to_bytes(lr1_pil), "input_lr1.png", "image/png")
                st.caption(f"LR1: {pixel_fingerprint(lr1_pil)}")
            with ex3:
                st.download_button("Download LR2", pil_to_bytes(lr2_pil), "input_lr2.png", "image/png")
                st.caption(f"LR2: {pixel_fingerprint(lr2_pil)}")

        st.divider()

        dual_ckpt_path   = Path(dual_ckpt_input)
        single_ckpt_path = Path(single_ckpt_input)

        if not single_ckpt_path.exists():
            st.warning(f"Single-EDSR checkpoint not found: `{single_ckpt_path}`")
        if not dual_ckpt_path.exists():
            st.warning(f"Dual-EDSR checkpoint not found: `{dual_ckpt_path}`")

        bc1, bc2 = st.columns(2)
        with bc1:
            run_single_flag = st.button(
                "Run Single-EDSR",
                disabled=not single_ckpt_path.exists(),
                use_container_width=True,
                key="btn_single",
            )
        with bc2:
            run_dual_flag = st.button(
                "Run Dual-EDSR",
                type="primary",
                disabled=not dual_ckpt_path.exists(),
                use_container_width=True,
                key="btn_dual",
            )

        for key in ("single_result", "dual_result", "prev_upload"):
            if key not in st.session_state:
                st.session_state[key] = None

        if st.session_state["prev_upload"] != uploaded.name:
            st.session_state["single_result"] = None
            st.session_state["dual_result"]   = None
            st.session_state["prev_upload"]   = uploaded.name

        if run_single_flag:
            with st.spinner("Running Single-EDSR…"):
                m  = load_single_model(str(single_ckpt_path), config, device_str)
                sr = run_single(m, lr1_pil, device_str)
                st.session_state["single_result"] = (sr, *compute_metrics(sr, hr_pil))

        if run_dual_flag:
            with st.spinner("Running Dual-EDSR…"):
                m  = load_dual_model(str(dual_ckpt_path), config, device_str)
                sr = run_dual(m, lr1_pil, lr2_pil, device_str)
                st.session_state["dual_result"] = (sr, *compute_metrics(sr, hr_pil))

        bic_pil            = bicubic_up(lr1_pil, scale=scale)
        psnr_bic, ssim_bic = compute_metrics(bic_pil, hr_pil)

        st.subheader("Results")

        have_single = st.session_state["single_result"] is not None
        have_dual   = st.session_state["dual_result"]   is not None
        num_cols    = 1 + int(have_single) + int(have_dual)
        cols        = st.columns(num_cols)
        col_i       = 0

        with cols[col_i]:
            st.image(bic_pil, caption="Bicubic (LR1 upscaled)", use_container_width=True)
            metric_html("Bicubic baseline", psnr_bic, ssim_bic)
        col_i += 1

        if have_single:
            sr, p, s = st.session_state["single_result"]
            with cols[col_i]:
                st.image(sr, caption="Single-EDSR (LR1 only)", use_container_width=True)
                metric_html("Single-EDSR", p, s, delta_p=p - psnr_bic)
            col_i += 1

        if have_dual:
            sr, p, s = st.session_state["dual_result"]
            with cols[col_i]:
                st.image(sr, caption="Dual-EDSR (LR1 + LR2)", use_container_width=True)
                metric_html("Dual-EDSR", p, s, delta_p=p - psnr_bic)

        if have_single and have_dual:
            import pandas as pd
            _, p_sgl, s_sgl   = st.session_state["single_result"]
            _, p_dual, s_dual = st.session_state["dual_result"]

            df = pd.DataFrame({
                "Method":    ["Bicubic", "Single-EDSR", "Dual-EDSR"],
                "PSNR (dB)": [f"{psnr_bic:.2f}", f"{p_sgl:.2f}", f"{p_dual:.2f}"],
                "SSIM":      [f"{ssim_bic:.4f}", f"{s_sgl:.4f}", f"{s_dual:.4f}"],
                "ΔPSNR":     ["—", f"{p_sgl-psnr_bic:+.2f}", f"{p_dual-psnr_bic:+.2f}"],
                "ΔSSIM":     ["—", f"{s_sgl-ssim_bic:+.4f}", f"{s_dual-ssim_bic:+.4f}"],
            })

            st.divider()
            st.subheader("Comparison Summary")
            st.dataframe(df, hide_index=True, use_container_width=True)

            st.subheader("Download Outputs")
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.download_button("Bicubic SR",     pil_to_bytes(bic_pil), "bicubic_sr.png",     "image/png")
            with dc2:
                sr_sgl, _, _ = st.session_state["single_result"]
                st.download_button("Single-EDSR SR", pil_to_bytes(sr_sgl),  "single_edsr_sr.png", "image/png")
            with dc3:
                sr_dual, _, _ = st.session_state["dual_result"]
                st.download_button("Dual-EDSR SR",   pil_to_bytes(sr_dual), "dual_edsr_sr.png",   "image/png")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Blind Evaluation (Sentinel-2 / no ground truth)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_blind:
    st.markdown(
        "Upload **two real LR images** of the same scene (e.g. two Sentinel-2 acquisitions "
        "on different dates). No ground truth needed — quality is assessed using "
        "**NIQE** and **BRISQUE** (lower = better)."
    )

    dual_ckpt_path_blind   = Path(dual_ckpt_input)
    single_ckpt_path_blind = Path(single_ckpt_input)

    with st.expander("Advanced options", expanded=False):
        include_single = st.checkbox(
            "Include Single-EDSR (LR1 only)",
            value=False,
            help="Optional: adds Single-EDSR to the blind comparison.",
        )

    if include_single and not single_ckpt_path_blind.exists():
        st.warning(f"Single-EDSR checkpoint not found: `{single_ckpt_path_blind}`")
    if not dual_ckpt_path_blind.exists():
        st.warning(f"Dual-EDSR checkpoint not found: `{dual_ckpt_path_blind}`")

    b1, b2 = st.columns(2)
    with b1:
        up_lr1 = st.file_uploader(
            "Upload LR1 (first acquisition)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="blind_lr1",
        )
    with b2:
        up_lr2 = st.file_uploader(
            "Upload LR2 (second acquisition)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="blind_lr2",
        )

    if up_lr1 is None or up_lr2 is None:
        st.info("Upload both LR images to start.")
    else:
        lr1_blind = Image.open(io.BytesIO(up_lr1.read())).convert("RGB")
        lr2_blind = Image.open(io.BytesIO(up_lr2.read())).convert("RGB")

        # Ensure same size — crop to minimum common size
        w = min(lr1_blind.width, lr2_blind.width)
        h = min(lr1_blind.height, lr2_blind.height)
        lr1_blind = lr1_blind.crop((0, 0, w, h))
        lr2_blind = lr2_blind.crop((0, 0, w, h))

        st.subheader("Uploaded LR Images")
        ci1, ci2 = st.columns(2)
        with ci1:
            st.image(lr1_blind, caption=f"LR1 — {up_lr1.name}", use_container_width=True)
        with ci2:
            st.image(lr2_blind, caption=f"LR2 — {up_lr2.name}", use_container_width=True)

        n1 = up_lr1.name.lower()
        n2 = up_lr2.name.lower()
        if any(tok in n1 or tok in n2 for tok in ("_sr", "bicubic_sr", "single_edsr", "dual_edsr")):
            st.warning("These filenames look like SR outputs. Blind mode expects true LR inputs; SR files will be upscaled again.")

        with st.expander("Diagnostics", expanded=False):
            st.caption(f"LR1: {pixel_fingerprint(lr1_blind)}")
            st.caption(f"LR2: {pixel_fingerprint(lr2_blind)}")
            st.caption(f"Expected SR output size: {lr1_blind.width * scale}×{lr1_blind.height * scale}")

        for key in ("blind_result", "blind_prev_pair"):
            if key not in st.session_state:
                st.session_state[key] = None

        curr_pair = (up_lr1.name, up_lr2.name)
        if st.session_state["blind_prev_pair"] != curr_pair:
            st.session_state["blind_result"] = None
            st.session_state["blind_prev_pair"] = curr_pair

        st.divider()

        can_run_single = include_single and single_ckpt_path_blind.exists()
        can_run_dual   = dual_ckpt_path_blind.exists()

        if st.button(
            "Run blind evaluation",
            type="primary",
            disabled=not (can_run_single or can_run_dual),
            use_container_width=True,
            key="btn_blind",
        ):
            with st.spinner("Running models and computing blind metrics…"):
                bic_blind = bicubic_up(lr1_blind, scale=scale)

                bic_t = pil_to_tensor(bic_blind)
                niqe_bic = float(calculate_niqe(bic_t[0].cpu()))
                brisq_bic = float(calculate_brisque(bic_t[0].cpu()))

                res: dict = {
                    "bic": bic_blind,
                    "niqe_bic": niqe_bic,
                    "brisq_bic": brisq_bic,
                }

                if can_run_single:
                    model_single_blind = load_single_model(str(single_ckpt_path_blind), config, device_str)
                    single_sr = run_single(model_single_blind, lr1_blind, device_str)
                    single_t = pil_to_tensor(single_sr)
                    res.update({
                        "single_sr": single_sr,
                        "niqe_single": float(calculate_niqe(single_t[0].cpu())),
                        "brisq_single": float(calculate_brisque(single_t[0].cpu())),
                    })

                if can_run_dual:
                    model_dual_blind = load_dual_model(str(dual_ckpt_path_blind), config, device_str)
                    dual_sr = run_dual(model_dual_blind, lr1_blind, lr2_blind, device_str)
                    dual_t = pil_to_tensor(dual_sr)
                    res.update({
                        "dual_sr": dual_sr,
                        "niqe_dual": float(calculate_niqe(dual_t[0].cpu())),
                        "brisq_dual": float(calculate_brisque(dual_t[0].cpu())),
                    })

                st.session_state["blind_result"] = res

        if st.session_state.get("blind_result"):
            res = st.session_state["blind_result"]

            have_single = include_single and "single_sr" in res
            have_dual   = "dual_sr" in res

            st.subheader("Results")

            num_cols = 1 + int(have_single) + int(have_dual)
            cols = st.columns(num_cols)
            col_i = 0

            with cols[col_i]:
                st.image(res["bic"], caption="Bicubic upscale of LR1", use_container_width=True)
                st.markdown(
                    f"""
                    <div class="sr-card">
                      <div class="sr-card-title">Bicubic baseline</div>
                      <div class="sr-row">
                        <div class="sr-kpi"><span class="sr-kpi-label">NIQE</span><b>{res['niqe_bic']:.3f}</b></div>
                        <div class="sr-kpi"><span class="sr-kpi-label">BRISQUE</span><b>{res['brisq_bic']:.3f}</b></div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            col_i += 1

            if have_single:
                dn = res["niqe_bic"] - res["niqe_single"]
                db = res["brisq_bic"] - res["brisq_single"]
                color_n = "#a6e3a1" if dn >= 0 else "#f38ba8"
                color_b = "#a6e3a1" if db >= 0 else "#f38ba8"
                with cols[col_i]:
                    st.image(res["single_sr"], caption="Single-EDSR (LR1 only)", use_container_width=True)
                    st.markdown(
                        f"""
                        <div class="sr-card">
                          <div class="sr-card-title">Single-EDSR</div>
                          <div class="sr-kpi">
                            <span class="sr-kpi-label">NIQE</span><b>{res['niqe_single']:.3f}</b>
                            <span class="sr-delta" style="color:{color_n}">(Δ{dn:+.3f} vs bicubic)</span>
                          </div>
                          <div class="sr-kpi">
                            <span class="sr-kpi-label">BRISQUE</span><b>{res['brisq_single']:.3f}</b>
                            <span class="sr-delta" style="color:{color_b}">(Δ{db:+.3f} vs bicubic)</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                col_i += 1

            if have_dual:
                dn = res["niqe_bic"] - res["niqe_dual"]
                db = res["brisq_bic"] - res["brisq_dual"]
                color_n = "#a6e3a1" if dn >= 0 else "#f38ba8"
                color_b = "#a6e3a1" if db >= 0 else "#f38ba8"
                with cols[col_i]:
                    st.image(res["dual_sr"], caption="Dual-EDSR (LR1 + LR2)", use_container_width=True)
                    st.markdown(
                        f"""
                        <div class="sr-card">
                          <div class="sr-card-title">Dual-EDSR</div>
                          <div class="sr-kpi">
                            <span class="sr-kpi-label">NIQE</span><b>{res['niqe_dual']:.3f}</b>
                            <span class="sr-delta" style="color:{color_n}">(Δ{dn:+.3f} vs bicubic)</span>
                          </div>
                          <div class="sr-kpi">
                            <span class="sr-kpi-label">BRISQUE</span><b>{res['brisq_dual']:.3f}</b>
                            <span class="sr-delta" style="color:{color_b}">(Δ{db:+.3f} vs bicubic)</span>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            import pandas as pd
            methods = ["Bicubic"]
            niqe_vals = [f"{res['niqe_bic']:.3f}"]
            brisq_vals = [f"{res['brisq_bic']:.3f}"]

            if have_single:
                methods.append("Single-EDSR")
                niqe_vals.append(f"{res['niqe_single']:.3f}")
                brisq_vals.append(f"{res['brisq_single']:.3f}")

            if have_dual:
                methods.append("Dual-EDSR (Ours)")
                niqe_vals.append(f"{res['niqe_dual']:.3f}")
                brisq_vals.append(f"{res['brisq_dual']:.3f}")

            df_blind = pd.DataFrame({
                "Method": methods,
                "NIQE ↓": niqe_vals,
                "BRISQUE ↓": brisq_vals,
            })

            st.divider()
            st.subheader("Blind Metrics Summary")
            st.dataframe(df_blind, hide_index=True, use_container_width=True)
            st.caption("Lower NIQE and BRISQUE indicate better perceptual quality.")

            st.subheader("Download Output")
            dl_cols = st.columns(len(methods))
            dl_i = 0
            with dl_cols[dl_i]:
                st.download_button("Bicubic SR", pil_to_bytes(res["bic"]), "blind_bicubic_sr.png", "image/png")
            dl_i += 1
            if have_single:
                with dl_cols[dl_i]:
                    st.download_button(
                        "Single-EDSR SR",
                        pil_to_bytes(res["single_sr"]),
                        "blind_single_edsr_sr.png",
                        "image/png",
                    )
                dl_i += 1
            if have_dual:
                with dl_cols[dl_i]:
                    st.download_button(
                        "Dual-EDSR SR",
                        pil_to_bytes(res["dual_sr"]),
                        "blind_dual_edsr_sr.png",
                        "image/png",
                    )

