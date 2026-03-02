# Dual-Image Super-Resolution for Remote Sensing Imagery

A dual-input super-resolution framework based on a modified **EDSR** architecture that fuses complementary information from two differently-degraded low-resolution images to reconstruct a high-resolution output at ×4 scale.

---

## Architecture

```
LR1 (bicubic ×4) ──→ ShallowConv ──┐
                                    ├──→ Concat → FusionConv(1×1) → 16 ResBlocks → PixelShuffle → HR
LR2 (blur+bic ×4) ─→ ShallowConv ──┘
```

| Component | Details |
|---|---|
| Residual blocks | 16 (no BN, residual scaling) |
| Feature channels | 64 |
| Fusion | Channel-wise concat + 1×1 conv |
| Upsampling | PixelShuffle ×4 |
| MeanShift | ImageNet RGB mean normalisation |

---

## Dataset — NWPU-RESISC45

31,500 remote sensing images · 45 scene categories · 256×256 RGB · loaded from HuggingFace (`timm/resisc45`)

| Split | Images |
|---|---|
| Train (70 %) | 22,050 |
| Val (15 %) | 4,725 |
| Test (15 %) | 4,725 |

**LR pair generation (on-the-fly):**
- **LR1** — bicubic downsampling ×4
- **LR2** — Gaussian blur (σ = 1.5) + bicubic downsampling ×4

---

## Results on NWPU-RESISC45 (×4)

| Method | Input | PSNR (dB) | SSIM |
|---|---|---|---|
| Bicubic | LR1 | — | — |
| Single-EDSR | LR1 | — | — |
| **Dual-EDSR** | LR1 + LR2 | — | — |

*Fill in after running `train.py` and `train_single.py`.*

---

## Quick Start

### 1. Install dependencies

```bash
cd src
pip install -r requirements.txt
```

### 2. Train

**Dual-EDSR** (downloads NWPU-RESISC45 ~600 MB on first run via HuggingFace):
```bash
python train.py
```

**Single-EDSR baseline** (same config, same hyperparams):
```bash
python train_single.py
```

Both scripts print best PSNR / SSIM at the end. Checkpoints saved to `src/checkpoints/`.

### 3. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/dual_edsr_best.pth
```

### 4. Web demo

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Upload any HR image — the app synthesises LR1/LR2, runs both models, and shows PSNR + SSIM side by side.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | NWPU-RESISC45 (HuggingFace `timm/resisc45`) |
| HR size | 256 × 256 |
| LR size | 64 × 64 (÷4) |
| Batch size | 16 |
| Epochs | 100 (early stopping, patience 20) |
| Optimiser | Adam, lr = 1×10⁻⁴, weight_decay = 0 |
| Loss | L1 (λ=1.0) + SSIM (λ=0.1) |
| LR scheduler | ReduceLROnPlateau ×0.5, patience 10 |
| Augmentation | H-flip, V-flip, 90° rotations |

---

## Project Structure

```
src/
├── models/
│   ├── dual_edsr.py       # DualEDSR (dual-input) + SingleEDSR (baseline)
│   └── blocks.py          # ResidualBlock, Upsampler, MeanShift
├── data/
│   ├── dataset.py         # DualSRDataset, DualSRHFDataset, get_hf_dataloaders
│   └── prepare_data.py    # Offline NWPU-RESISC45 preparation (Kaggle / Drive)
├── utils/
│   ├── losses.py          # CombinedLoss (L1 + SSIM)
│   └── metrics.py         # PSNR, SSIM, NIQE, BRISQUE
├── config/
│   └── config.yaml        # All hyperparameters
├── train.py               # Dual-EDSR training
├── train_single.py        # Single-EDSR baseline training
├── evaluate.py            # Full-reference + blind evaluation
├── inference.py           # Single-image inference
├── app.py                 # Streamlit evaluation demo
└── requirements.txt
```

---

## Checkpoint Paths

| File | Saved by |
|---|---|
| `checkpoints/dual_edsr_best.pth` | `train.py` |
| `checkpoints/single_edsr_best.pth` | `train_single.py` |

---

## Offline Dataset Preparation (optional)

Only needed if you prefer not to use HuggingFace streaming.

```bash
# Option A — Kaggle (already downloaded)
unzip downloads/nwpu-resisc45.zip -d downloads/
python src/data/prepare_data.py --raw_dir downloads/NWPU-RESISC45

# Then train without HF:
python src/train.py --no_hf --dataset_dir dataset
```

---

## References

- Lim et al., *Enhanced Deep Residual Networks for Single Image Super-Resolution*, CVPRW 2017
- Cheng et al., *Remote Sensing Image Scene Classification: Benchmark and State of the Art*, Proceedings of the IEEE 2017
