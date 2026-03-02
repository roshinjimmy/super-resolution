# 🔬 Dual-SR: Dual-Image Super-Resolution for Satellite Imagery

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning approach to **4× super-resolution** using **dual low-resolution inputs** with different degradation patterns. This project demonstrates that combining information from two differently degraded images improves reconstruction quality compared to single-input methods.

![Demo](assets/demo.png)

---

## 🎯 Key Results

| Model | Inputs | PSNR (dB) | Improvement |
|-------|--------|-----------|-------------|
| **DualEDSR** | LR1 + LR2 | **27.95** | **+1.46 dB** ✅ |
| SingleEDSR | LR1 only | 26.49 | baseline |
| Bicubic | - | 25.24 | - |

> **✅ DualEDSR achieves 1.46 dB higher PSNR than SingleEDSR**, proving that dual-input fusion provides complementary information for better super-resolution!

---

## 🏗️ Architecture

### Dual-Branch EDSR Architecture

```
┌─────────────┐                    ┌─────────────┐
│   LR1 (4×)  │                    │   LR2 (4×)  │
│  (Bicubic)  │                    │ (Blur+Bic)  │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       ▼                                  ▼
┌─────────────┐                    ┌─────────────┐
│  Shallow    │                    │  Shallow    │
│   Conv      │                    │   Conv      │
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       └──────────────┬───────────────────┘
                      ▼
              ┌───────────────┐
              │   Concat      │
              │   (128 ch)    │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │  Fusion Conv  │
              │   (64 ch)     │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │  16 Residual  │
              │    Blocks     │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │  PixelShuffle │
              │  Upsampling   │
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │   HR Output   │
              │   (256×256)   │
              └───────────────┘
```

### Dual Input Strategy

| Input | Degradation | Characteristics |
|-------|-------------|-----------------|
| **LR1** | Bicubic 4× | Sharp edges, aliasing artifacts |
| **LR2** | Gaussian blur (σ=1.5) + Bicubic 4× | Smooth, less aliasing |

The model learns to fuse complementary information from both degradations for optimal reconstruction.

---

## 📁 Project Structure

```
dual-sr/
├── models/
│   ├── dual_edsr.py      # DualEDSR & SingleEDSR models
│   └── blocks.py         # Residual blocks, MeanShift, Upsampler
├── data/
│   ├── dataset.py        # PyTorch dataset classes
│   ├── prepare_data.py   # Data preparation scripts
│   └── prepare_kaggle_data.py  # UC Merced dataset prep
├── utils/
│   ├── losses.py         # L1 + SSIM combined loss
│   └── metrics.py        # PSNR, SSIM calculations
├── config/
│   └── config.yaml       # Training configuration
├── train.py              # DualEDSR training script
├── train_single.py       # SingleEDSR baseline training
├── evaluate.py           # Model evaluation
├── inference.py          # Single image inference
├── app.py                # Streamlit web demo
└── requirements.txt      # Dependencies
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/raunzw/dual-sr.git
cd dual-sr
pip install -r requirements.txt
```

### 2. Prepare Dataset

Download the [UC Merced Land Use dataset](https://www.kaggle.com/datasets/apollo2506/landuse-scene-classification) and prepare:

```bash
python data/prepare_kaggle_data.py --source_dir path/to/ucmerced --output_dir dataset
```

### 3. Train Models

**DualEDSR (Dual-Input):**
```bash
python train.py --config config/config.yaml --dataset_dir dataset --epochs 50
```

**SingleEDSR (Baseline):**
```bash
python train_single.py --dataset_dir dataset --epochs 50
```

### 4. Run Web Demo

```bash
python -m streamlit run app.py
```
Open http://localhost:8502 in your browser.

---

## 🎮 Web Demo Features

The Streamlit demo provides:

1. **Upload HR Image** - Drag and drop any image
2. **Auto-generate LR pairs** - Creates LR1 (bicubic) and LR2 (blur+bicubic)
3. **Run Super-Resolution** - Process with both models
4. **View Metrics** - Compare PSNR values side-by-side
5. **Interactive Comparison** - View all outputs together

---

## 📊 Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | UC Merced (2100 images) |
| Split | Train: 1570, Val: 335, Test: 335 |
| HR Size | 256 × 256 |
| LR Size | 64 × 64 |
| Scale Factor | 4× |
| Epochs | 50 |
| Batch Size | 4 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Loss | L1 |

---

## 📈 Performance Comparison

### Validation PSNR over Epochs

| Epoch | DualEDSR | SingleEDSR |
|-------|----------|------------|
| 10 | 26.12 dB | 25.31 dB |
| 20 | 27.05 dB | 25.89 dB |
| 30 | 27.48 dB | 26.21 dB |
| 40 | 27.78 dB | 26.39 dB |
| **50** | **27.95 dB** | **26.49 dB** |

---

## 🔧 Model Checkpoints

After training, checkpoints are saved in:
- `checkpoints/best_model.pth` - DualEDSR
- `checkpoints_single/best_model.pth` - SingleEDSR

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{dual_sr_2024,
  author = {Nandu Prakash},
  title = {Dual-SR: Dual-Image Super-Resolution for Satellite Imagery},
  year = {2024},
  url = {https://github.com/raunzw/dual-sr}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [EDSR Paper](https://arxiv.org/abs/1707.02921) - Enhanced Deep Residual Networks for Single Image Super-Resolution
- [UC Merced Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) - Land Use Scene Classification
- [PyTorch](https://pytorch.org/) - Deep Learning Framework

---

<p align="center">Made with ❤️ for satellite imagery enhancement</p>
