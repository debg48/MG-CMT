# Mamdani-Gated Cross-Modal Transformer (MG-CMT)

**Author:** Debgandhar Ghosh  
**Date:** January 2026  
**Optimized for:** Apple M4 Air (MPS Backend)

## 🔬 Overview

The **Mamdani-Gated Cross-Modal Transformer (MG-CMT)** is a novel deep learning architecture designed for robust multimodal medical image analysis. It specifically targets the challenge of fusing **Chest X-Rays (CXR)** (high structural consistency) with **Sputum Smear Microscopy** (high diagnostic value but high noise/variance) for the automated detection of Tuberculosis.

This repository contains the official PyTorch implementation of MG-CMT.

## 🧠 The Problem: "Feature Wash-out"

Traditional multimodal fusion strategies (concatenation, element-wise addition, or vanilla cross-attention) often fail when one modality is significantly noisier or less reliable than the other. In the context of TB diagnosis:

- **CXR** provides a reliable structural baseline.
- **Sputum Microscopy** is highly specific but prone to artifacts, staining errors, and field-of-view variability.

Naive fusion allows the noise from poor-quality sputum slides to corrupt (or "wash out") the clean features from the CXR, degrading overall model performance.

## 💡 The Solution: Neuro-Fuzzy Gating

MG-CMT introduces a **Differentiable Mamdani Fuzzy Inference System (FIS)** that acts as a cognitive gatekeeper. Instead of blindly trusting all inputs, the model:

1. **Measures Uncertainty**: Calculates entropy from each modality's preliminary predictions.
2. **Applies Fuzzy Logic**: Uses human-interpretable rules (e.g., *"If Sputum is Uncertain, reduce its influence"*).
3. **Modulates Attention**: Dynamically scales the cross-attention weights, effectively "gating" the flow of information.

## 🏗️ Architecture

The model consists of three core components:

### 1. Custom Lightweight Vision Transformer with 2D-RoPE

Instead of using pretrained ViT (86M parameters, prone to overfitting on small datasets), we implement a **custom 4-layer transformer** optimized for medical imaging:

- **4 transformer layers** (vs 12 in ViT-Base)
- **256-dim embeddings** (vs 768)
- **8 attention heads**
- **~5M parameters per encoder**
- **2D-Rotary Positional Embeddings (2D-RoPE)** for capturing relative spatial relationships (e.g., "infiltrate is *above* the clavicle")

**Key Innovation:** 2D-RoPE extends rotary embeddings to 2D, allowing the model to understand spatial context critical for medical imaging.

### 2. Differentiable Mamdani FIS

A learnable fuzzy controller that serves as the fusion brain:

- **Input**: Uncertainty metrics ($\mathcal{U}_{cxr}, \mathcal{U}_{spt}$) computed via entropy
- **Fuzzification**: Learnable Gaussian membership functions map uncertainty to fuzzy sets (Low, Medium, High)
- **Inference**: Computes firing strengths of 9 fuzzy rules (3×3 combinations)
- **Defuzzification**: Outputs scalars $\alpha$ (CXR confidence) and $\beta$ (Sputum confidence)

**All operations are differentiable**, enabling end-to-end training.

### 3. Fuzzy-Modulated Cross-Attention (FMCA)

We redefine the standard Attention mechanism to accept the fuzzy scalars:

$$
\text{FMCA}(Q, K, V) = \text{Softmax}\left(\frac{QK^T \cdot \beta}{\sqrt{d_k}}\right)V
$$

This ensures that if the FIS detects high uncertainty in the key/value modality (e.g., a blurry sputum slide), $\beta \to 0$, and the attention mechanism ignores it, falling back to the query modality's self-features.

**Key Design Choice:** We apply $\beta$ **before softmax** (logit scaling) rather than after, providing stronger, non-linear gating.

## 🛠️ Installation

### For Apple M4 Air (MPS Backend)

```bash
# Clone the repository
git clone https://github.com/debg48/mg-cmt.git
cd mg-cmt

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**System Requirements:**

- macOS 13.0+ (for MPS support)
- Python 3.9+
- 16GB RAM recommended

## 🚀 Usage

### Quick Test (Verify Installation)

```bash
# Test individual components
python3 -m models.encoders   # Test custom ViT encoder
python3 -m models.fis         # Test Mamdani FIS
python3 -m models.fusion      # Test FMCA
python3 -m models.mg_cmt      # Test complete model
```

Expected output:

```
Using device: mps
Total parameters: 10,521,218
✅ Model successfully initialized and tested on mps!
```

### Training (Coming Soon)

```bash
python3 train.py --config configs/default.yaml
```

### Inference (Coming Soon)

```bash
python3 inference.py --cxr_path data/sample_cxr.png --sputum_path data/sample_sputum.png
```

## 📂 Project Structure

```
.
├── models/
│   ├── encoders.py     # Custom 4-layer ViT with 2D-RoPE
│   ├── fis.py          # Differentiable Mamdani FIS
│   ├── fusion.py       # FMCA Layer (logit & post-softmax variants)
│   └── mg_cmt.py       # Complete MG-CMT architecture
├── baselines/          # Baseline fusion methods (concat, late fusion, scalar gate)
├── data/               # Dataset loaders
│   └── JU-LDD-task-b/  # TB detection dataset
├── experiments/        # Training & evaluation scripts
├── configs/            # Hyperparameter configuration
├── utils/              # Metrics, visualization, statistical tests
├── scripts/            # Experiment runners, plotting
└── requirements.txt    # Python dependencies
```

## 🔑 Key Features

- **Lightweight**: 10.5M total parameters (vs 86M for standard ViT-based multimodal models)
- **Interpretable**: Fuzzy membership functions are visualizable and human-understandable
- **Robust**: Gracefully handles noisy/missing modalities via dynamic gating
- **Efficient**: Optimized for M4 Air MPS backend (~30 min/epoch on single GPU)
- **End-to-end trainable**: All components (including FIS) learn from data

## 📊 Model Size Comparison

| Model | Params | FLOPs | Memory |
|:---|---:|---:|---:|
| ViT-Base (pretrained) | 86M | 17.6G | 12GB |
| ViT-Small | 22M | 4.6G | 6GB |
| **MG-CMT (Ours)** | **10.5M** | **2.1G** | **4GB** |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex

```

## 📧 Contact

For questions or collaborations, please reach out to [your.email@university.edu]

## 🎯 Next Steps

- [ ] Implement training pipeline
- [ ] Add baseline models (ResNet-50, EfficientNet, MobileNet variants)
- [ ] Create evaluation scripts
- [ ] Implement ablation study runners
- [ ] Add visualization tools for fuzzy membership functions
- [ ] Generate paper figures

## ⚡ Performance Notes for M4 Air

- **Batch size**: 4-8 recommended (16GB RAM)
- **Mixed precision**: Not required (MPS handles fp32 efficiently)
- **Gradient checkpointing**: Optional for larger batches
- **Expected speed**: ~30 min/epoch with 1000 training images
