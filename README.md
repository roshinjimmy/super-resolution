# Dual-Image Super-Resolution for Remote Sensing Imagery

A dual-input super-resolution framework based on a modified **EDSR** architecture that fuses complementary information from two differently-degraded low-resolution images to reconstruct a high-resolution output at ×4 scale.

---

## Architecture

```
LR1 (bicubic ×4) ──→ ShallowConv(3×3) ──┐
                                          ├──→ Concat → FusionConv(1×1) → 16 ResBlocks → PixelShuffle(×4) → ReconConv(3×3) → HR
LR2 (blur+bic ×4) ─→ ShallowConv(3×3) ──┘
```

| Component | Details |
|---|---|
| Residual blocks | 16 (no BatchNorm, res\_scale = 0.1) |
| Feature channels | 64 |
| Fusion | Channel-wise concat + 1×1 conv |
| Upsampling | PixelShuffle ×4 |
| Normalisation | MeanShift (ImageNet RGB mean) |
| Training | AMP (mixed precision), gradient clipping (norm 1.0) |

---

## Dataset — NWPU-RESISC45

31,500 remote sensing images · 45 scene categories · 256×256 RGB · loaded from HuggingFace (`timm/resisc45`)

| Split | Images |
|---|---|
| Train (70 %) | 22,050 |
| Val (15 %) | 4,725 |
| Test (15 %) | 4,725 |

**LR pair generation (on-the-fly, seed = 42):**
- **LR1** — bicubic downsampling ×4
- **LR2** — Gaussian blur (σ = 1.5) + bicubic downsampling ×4

---

## Results on NWPU-RESISC45 (×4)

| Method | Input | PSNR (dB) | SSIM |
|---|---|---|---|
| Bicubic | LR1 | — | — |
| Single-EDSR | LR1 | — | — |
| **Dual-EDSR** | LR1 + LR2 | — | — |

*Fill in after running `evaluate.py --checkpoint checkpoints/dual_edsr_best.pth --single_checkpoint checkpoints/single_edsr_best.pth`.*

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train

See the [Training](#training) section below for full details.

```bash
cd src

# Dual-EDSR (primary model)
python train.py --config config/config.yaml

# Single-EDSR baseline
python train_single.py --config config/config.yaml
```

Both scripts download NWPU-RESISC45 (~600 MB) from HuggingFace on first run.  
Checkpoints are saved to `src/checkpoints/` every 10 epochs and whenever a new best PSNR is reached.

### 3. Evaluate

```bash
cd src
# Table 1 — all three rows (Bicubic, Single-EDSR, Dual-EDSR)
python evaluate.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --single_checkpoint checkpoints/single_edsr_best.pth
```

### 4. Web demo

```bash
cd src
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Upload any HR image — the app synthesises LR1/LR2, runs both models, and shows PSNR + SSIM side by side.

---

## Training

All hyperparameters live in `src/config/config.yaml`. CLI flags override config values when supplied.

### Dual-EDSR

```bash
cd src

# Standard run (HuggingFace dataset, auto GPU)
python train.py --config config/config.yaml

# Override epochs and batch size
python train.py --config config/config.yaml --epochs 50 --batch_size 8

# Force CPU
python train.py --config config/config.yaml --device cpu

# Use pre-downloaded dataset on disk instead of HuggingFace
python train.py --config config/config.yaml --no_hf --dataset_dir /path/to/dataset

# Resume from a checkpoint (picks up epoch, optimizer, scheduler, best metrics)
python train.py --config config/config.yaml --resume checkpoints/checkpoint_epoch_30.pth
python train.py --config config/config.yaml --resume checkpoints/dual_edsr_best.pth
```

**CLI reference — `train.py`**

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/config.yaml` | Path to YAML config |
| `--epochs` | *(from config)* | Override number of epochs |
| `--batch_size` | *(from config)* | Override batch size |
| `--resume` | — | Path to checkpoint to resume from |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |
| `--use_hf` | *(from config)* | Force HuggingFace loading |
| `--no_hf` | — | Force disk-based loading |
| `--dataset_dir` | `dataset` | Root dir for disk dataset |

---

### Single-EDSR baseline

```bash
cd src

# Standard run
python train_single.py --config config/config.yaml

# Override epochs
python train_single.py --config config/config.yaml --epochs 50

# Resume
python train_single.py --config config/config.yaml --resume checkpoints/single_checkpoint_epoch_30.pth
python train_single.py --config config/config.yaml --resume checkpoints/single_edsr_best.pth
```

**CLI reference — `train_single.py`**

| Flag | Default | Description |
|---|---|---|
| `--config` | `config/config.yaml` | Path to YAML config |
| `--epochs` | *(from config)* | Override number of epochs |
| `--resume` | — | Path to checkpoint to resume from |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |
| `--use_hf` | *(from config)* | Force HuggingFace loading |
| `--no_hf` | — | Force disk-based loading |
| `--dataset_dir` | `dataset` | Root dir for disk dataset |

---

### Stopping and resuming

You can interrupt training at any time with **Ctrl+C**. Checkpoints are written every 10 epochs (configurable via `checkpoint.save_every` in config) and on every new best-PSNR epoch. Resume with `--resume`:

```bash
# Resume Dual-EDSR from latest periodic checkpoint
python train.py --resume checkpoints/checkpoint_epoch_40.pth

# Resume Single-EDSR from best checkpoint
python train_single.py --resume checkpoints/single_edsr_best.pth
```

Resumed state includes: model weights, optimizer, LR scheduler, GradScaler (AMP), best PSNR/SSIM, and early-stopping counter.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Dataset | NWPU-RESISC45 (HuggingFace `timm/resisc45`) |
| HR size | 256 × 256 |
| LR size | 64 × 64 (÷4) |
| Batch size | 16 |
| Epochs | 100 (early stopping, patience 20) |
| Optimiser | Adam, lr = 1×10⁻⁴, weight\_decay = 0 |
| Loss | L1 (λ=1.0) + SSIM (λ=0.1) |
| LR scheduler | ReduceLROnPlateau ×0.5, patience 10, min 1×10⁻⁶ |
| Gradient clipping | max norm = 1.0 |
| Mixed precision | AMP enabled (`torch.amp.autocast` + `GradScaler`) |
| Residual scaling | res\_scale = 0.1 |
| Augmentation | H-flip, V-flip, 90° rotations |
| Seed | 42 |

---

## Evaluation

```bash
cd src

# Full-reference evaluation on the test split (produces Table 1)
python evaluate.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --single_checkpoint checkpoints/single_edsr_best.pth

# Save output images as well
python evaluate.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --single_checkpoint checkpoints/single_edsr_best.pth \
    --output_dir outputs/test_results

# Blind metrics only (no HR reference needed)
python evaluate.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --blind

# Evaluate on custom LR image directories
python evaluate.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --lr1_dir /path/to/lr1 --lr2_dir /path/to/lr2
```

**CLI reference — `evaluate.py`**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to DualEDSR checkpoint |
| `--single_checkpoint` | — | Path to SingleEDSR checkpoint (adds Table 1 row) |
| `--config` | `config/config.yaml` | Path to YAML config |
| `--output_dir` | `outputs` | Directory to write result images |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |
| `--use_hf` / `--no_hf` | *(from config)* | Dataset source override |
| `--blind` | — | Compute blind metrics (NIQE, BRISQUE) |
| `--lr1_dir` | — | Custom LR1 image directory |
| `--lr2_dir` | — | Custom LR2 image directory |

---

## Sentinel-2 Inference

For real Sentinel-2 tiles (no HR reference), using overlapping tile inference (64×64 LR tiles, 16 px overlap, linear ramp blending):

```bash
cd src

python inference.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --input_dir /path/to/sentinel2/bands \
    --output_dir outputs/sentinel \
    --method native

# Synthetic evaluation (downscale the input to create a pseudo-LR, compare with original)
python inference.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --input_dir /path/to/sentinel2/bands \
    --method synthetic

# Custom tile size and overlap
python inference.py \
    --checkpoint checkpoints/dual_edsr_best.pth \
    --input_dir /path/to/sentinel2/bands \
    --tile_size 64 --overlap 16
```

**CLI reference — `inference.py`**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to DualEDSR checkpoint |
| `--input_dir` | *(required)* | Directory containing Sentinel-2 band images |
| `--config` | `config/config.yaml` | Path to YAML config |
| `--output_dir` | `outputs/sentinel` | Output directory |
| `--method` | `native` | `native` (real SR) or `synthetic` (pseudo-LR evaluation) |
| `--tile_size` | `64` | LR tile size in pixels |
| `--overlap` | `16` | Overlap between tiles in LR pixels |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |

---

## Project Structure

```
src/
├── models/
│   ├── dual_edsr.py       # DualEDSR (dual-input) + SingleEDSR (baseline)
│   └── blocks.py          # ResidualBlock, Upsampler, MeanShift
├── data/
│   ├── dataset.py         # DualSRHFDataset, DualSRDataset, get_hf_dataloaders
│   └── prepare_data.py    # Offline NWPU-RESISC45 preparation (Kaggle / Drive)
├── utils/
│   ├── losses.py          # CombinedLoss (L1 + SSIM)
│   └── metrics.py         # PSNR, SSIM, NIQE, BRISQUE
├── config/
│   └── config.yaml        # All hyperparameters
├── train.py               # Dual-EDSR training (with resume support)
├── train_single.py        # Single-EDSR baseline training (with resume support)
├── evaluate.py            # Full-reference + blind evaluation (Table 1)
├── inference.py           # Sentinel-2 tile inference
├── app.py                 # Streamlit evaluation demo
└── requirements.txt
```

---

## Checkpoint Paths

| File | Saved by | Contents |
|---|---|---|
| `checkpoints/dual_edsr_best.pth` | `train.py` | Best val-PSNR weights + full training state |
| `checkpoints/checkpoint_epoch_N.pth` | `train.py` | Periodic checkpoint every 10 epochs |
| `checkpoints/final_model.pth` | `train.py` | Weights at end of training |
| `checkpoints/single_edsr_best.pth` | `train_single.py` | Best val-PSNR weights + full training state |
| `checkpoints/single_checkpoint_epoch_N.pth` | `train_single.py` | Periodic checkpoint every 10 epochs |

---

## Offline Dataset Preparation (optional)

Only needed if you prefer not to use HuggingFace streaming.

```bash
# Option A — Kaggle
unzip downloads/nwpu-resisc45.zip -d downloads/
python src/data/prepare_data.py --raw_dir downloads/NWPU-RESISC45

# Train without HuggingFace
cd src
python train.py --no_hf --dataset_dir dataset
python train_single.py --no_hf --dataset_dir dataset
```

---

## References

- Lim et al., *Enhanced Deep Residual Networks for Single Image Super-Resolution*, CVPRW 2017
- Cheng et al., *Remote Sensing Image Scene Classification: Benchmark and State of the Art*, Proceedings of the IEEE 2017
