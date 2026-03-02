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
from utils.metrics import calculate_psnr, calculate_ssim

# ── constants ───────────────────────────────────────────────────────────────
CONFIG_PATH         = _HERE / "config" / "config.yaml"
DEFAULT_DUAL_CKPT   = _HERE / "checkpoints" / "dual_edsr_best.pth"
DEFAULT_SINGLE_CKPT = _HERE / "checkpoints" / "single_edsr_best.pth"


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
        res_scale=cfg.get("res_scale", 1.0),
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
        res_scale=cfg.get("res_scale", 1.0),
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
    delta_str = ""
    if delta_p is not None:
        color = "#a6e3a1" if delta_p >= 0 else "#f38ba8"
        sign  = "+" if delta_p >= 0 else ""
        delta_str = f'&nbsp; <span style="color:{color};font-size:0.85em">({sign}{delta_p:.2f} dB vs bicubic)</span>'
    st.markdown(
        f"""<div style="background:#1e1e2e;border-radius:8px;padding:10px 16px;margin-top:4px">
        <b>{label}</b>{delta_str}<br/>
        <span style="color:#a6e3a1">PSNR <b>{psnr:.2f} dB</b></span>
        &nbsp;&nbsp;
        <span style="color:#89dceb">SSIM <b>{ssim:.4f}</b></span>
        </div>""",
        unsafe_allow_html=True,
    )


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── page setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dual-SR Evaluation",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Dual-Image Super-Resolution — Evaluation Demo")
st.caption(
    "Upload a high-resolution image. The app synthesises LR1 (bicubic ÷4) and "
    "LR2 (Gaussian blur + bicubic ÷4), then runs Single-EDSR and Dual-EDSR for comparison."
)

config      = load_config()
scale       = config["model"].get("scale", 4)
blur_sigma  = config["data"].get("blur_sigma", 1.5)
device_str  = "cuda" if torch.cuda.is_available() else "cpu"

# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Checkpoints")
    dual_ckpt_input   = st.text_input("Dual-EDSR checkpoint",   value=str(DEFAULT_DUAL_CKPT))
    single_ckpt_input = st.text_input("Single-EDSR checkpoint", value=str(DEFAULT_SINGLE_CKPT))

    st.subheader("Degradation")
    blur_sigma = st.slider("LR2 Gaussian σ", 0.5, 5.0, float(blur_sigma), 0.1,
                           help="Must match the sigma used during training (default 1.5)")

    st.subheader("Device")
    st.info(f"**{device_str.upper()}**")

    st.markdown("---")
    st.caption(
        "Checkpoint files default to `src/checkpoints/`.  \n"
        "Run `train.py` and `train_single.py` first to generate them."
    )

# ── file upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload HR image (PNG / JPG / TIFF — treated as ground truth)",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded is None:
    st.info("👆 Upload an image to start.")
    st.stop()

# ── load & crop HR ─────────────────────────────────────────────────────────
hr_pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
w, h   = hr_pil.size
w_crop = (w // scale) * scale
h_crop = (h // scale) * scale
hr_pil = hr_pil.crop((0, 0, w_crop, h_crop))

st.success(
    f"Loaded **{uploaded.name}** — {w_crop}×{h_crop} px  →  "
    f"LR size: {w_crop//scale}×{h_crop//scale} px  (×{scale})"
)

# ── synthesise LR pair ───────────────────────────────────────────────────────
lr1_pil, lr2_pil = synthesise_lr_pair(hr_pil, scale=scale, blur_sigma=blur_sigma)

# ── show inputs ──────────────────────────────────────────────────────────────
st.subheader("Inputs")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(hr_pil,  caption="HR — ground truth", use_container_width=True)
with c2:
    st.image(lr1_pil, caption=f"LR1 — bicubic ÷{scale}", use_container_width=True)
with c3:
    st.image(lr2_pil, caption=f"LR2 — Gaussian(σ={blur_sigma:.1f}) + bicubic ÷{scale}",
             use_container_width=True)

st.markdown("---")

# ── checkpoint validation + run buttons ──────────────────────────────────────
dual_ckpt_path   = Path(dual_ckpt_input)
single_ckpt_path = Path(single_ckpt_input)

if not single_ckpt_path.exists():
    st.warning(f"Single-EDSR checkpoint not found: `{single_ckpt_path}`")
if not dual_ckpt_path.exists():
    st.warning(f"Dual-EDSR checkpoint not found: `{dual_ckpt_path}`")

bc1, bc2 = st.columns(2)
with bc1:
    run_single_flag = st.button(
        "▶ Run Single-EDSR",
        disabled=not single_ckpt_path.exists(),
        use_container_width=True,
    )
with bc2:
    run_dual_flag = st.button(
        "▶ Run Dual-EDSR",
        disabled=not dual_ckpt_path.exists(),
        use_container_width=True,
    )

# ── session state ─────────────────────────────────────────────────────────────
for key in ("single_result", "dual_result", "prev_upload"):
    if key not in st.session_state:
        st.session_state[key] = None

# Reset when new image is uploaded
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

# ── bicubic baseline (always shown) ──────────────────────────────────────────
bic_pil             = bicubic_up(lr1_pil, scale=scale)
psnr_bic, ssim_bic  = compute_metrics(bic_pil, hr_pil)

# ── results panel ─────────────────────────────────────────────────────────────
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

# ── summary table & downloads (once both models have run) ────────────────────
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

    st.markdown("---")
    st.subheader("📊 Comparison Summary")
    st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("⬇️ Download Outputs")
    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        st.download_button("Bicubic SR",     pil_to_bytes(bic_pil), "bicubic_sr.png",      "image/png")
    with dc2:
        sr_sgl, _, _ = st.session_state["single_result"]
        st.download_button("Single-EDSR SR", pil_to_bytes(sr_sgl),  "single_edsr_sr.png",  "image/png")
    with dc3:
        sr_dual, _, _ = st.session_state["dual_result"]
        st.download_button("Dual-EDSR SR",   pil_to_bytes(sr_dual), "dual_edsr_sr.png",    "image/png")
