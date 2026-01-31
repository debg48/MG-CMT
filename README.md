# MGM-TB-Former: Mamdani-Gated Multimodal Transformer (MGM-TB-Former)

**Author:** Debgandhar Ghosh  
**Date:** January 2026  
**Optimized for:** Apple M4 Air (MPS Backend)

## 🔬 Overview

**MGM-TB-Former** (formerly MG-CMT) is a novel **Hybrid Convolutional-Transformer** architecture designed for robust multimodal medical image analysis. It specifically targets the challenge of fusing **Chest X-Rays (CXR)** (high structural consistency) with **Sputum Smear Microscopy** (high diagnostic value but high noise/variance) for the automated detection of Tuberculosis.

This model leverages a **Convolutional Stem** for local feature extraction and a **Vision Transformer Body** for global context, modulated by a **Fuzzy Inference System** to handle uncertainty.

This repository contains the official PyTorch implementation of MGM-TB-Former.

## 🧠 The Problem: "Feature Wash-out"

Traditional multimodal fusion strategies (concatenation, element-wise addition, or vanilla cross-attention) often fail when one modality is significantly noisier or less reliable than the other. In the context of TB diagnosis:

- **CXR** provides a reliable structural baseline.
- **Sputum Microscopy** is highly specific but prone to artifacts, staining errors, and field-of-view variability.

Naive fusion allows the noise from poor-quality sputum slides to corrupt (or "wash out") the clean features from the CXR, degrading overall model performance.

## 💡 The Solution: Neuro-Fuzzy Gating

MGM-TB-Former introduces a **Differentiable Mamdani Fuzzy Inference System (FIS)** that acts as a cognitive gatekeeper. Instead of blindly trusting all inputs, the model:

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

### 3. Fuzzy-Modulated Cross-Attention (FMCA) with Gated Residual Fusion

We redefine the standard Attention mechanism to accept the fuzzy scalars with a **critical residual connection**:

```python
fused = CXR_features + β × CrossAttention(CXR_queries, Sputum_keys)
```

**Post-Softmax Gating:**
$$
\text{FMCA}(Q, K, V) = \beta \cdot \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**Why This Design?**

- **Residual Path:** If the FIS detects high uncertainty in Sputum ($\beta \to 0$), the cross-attention output is suppressed, but the model **falls back to clean CXR features** via the residual connection. Without this, $\beta \to 0$ would result in zero features!
- **Post-Softmax Scaling:** We apply $\beta$ **after softmax** to directly scale the magnitude of the attended features. Pre-softmax scaling would only flatten the distribution toward uniformity, not shut it off.

## 🛠️ Installation

### For Apple M4 Air (MPS Backend)

```bash
# Clone the repository
git clone https://github.com/debg48/mgm-tb-former.git
cd mgm-tb-former

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

### Quick Start

```bash
# 1. Set up environment (first time only)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Train MGM-TB-Former
python3 train.py --config configs/mgm_tb_former.yaml
```

### Train Individual Models

```bash
# Your model
python3 train.py --config configs/mgm_tb_former.yaml

# Baselines
python3 train.py --config configs/cxr_only.yaml
python3 train.py --config configs/sputum_only.yaml
python3 train.py --config configs/concat_fusion.yaml
python3 train.py --config configs/vanilla_cmt.yaml
python3 train.py --config configs/scalar_gate_mlp.yaml
```

### Override Settings

```bash
python3 train.py --config configs/mgm_tb_former.yaml --epochs 100 --batch_size 8
```

### Monitor Training

```bash
tensorboard --logdir checkpoints
# Open http://localhost:6006
```

## 📊 Analysis & Visualization

After training the model, run the analysis script to generate "High Gain" paper figures:

```bash
python3 analyze_robustness.py
```

This will generate the following plots in the `results/` directory:

1. **`failure_case_viz.png`**: Visualizes how the model handles sensor failure (Noisy CXR + Clean Sputum).
2. **`missing_modality.png`**: Benchmarks performance when one modality is completely missing (Graceful Degradation).

**Output:** Each training run creates `checkpoints/{model}_{timestamp}/` with plots (accuracy, loss, confusion matrix), checkpoints, and test results.

## 📂 Project Structure

```
.
├── models/
│   ├── encoders.py     # Custom 4-layer ViT with 2D-RoPE
│   ├── fis.py          # Differentiable Mamdani FIS
│   ├── fusion.py       # FMCA Layer (logit & post-softmax variants)
│   └── mgm_tb_former.py # Complete MGM-TB-Former architecture
├── baselines/          # Baseline fusion methods (concat, late fusion, scalar gate)
├── data/               # Dataset loaders
│   └── JU-LDD-task-b/  # TB detection dataset
├── experiments/        # Training & evaluation scripts
├── configs/            # Hyperparameter configuration
│   └── mgm_tb_former.yaml
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
| **MGM-TB-Former (Ours)** | **10.5M** | **2.1G** | **4GB** |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex

```

## 📧 Contact

For questions or collaborations, please reach out to [your.email@university.edu]

## 🎯 Experimental Setup

### Fair Benchmarking

All models use **IDENTICAL** hyperparameters to ensure fair comparison:

- **Architecture**: 4 layers, 256-dim embeddings, 8 attention heads
- **Training**: 50 epochs, batch size 4, learning rate 1e-4
- **Optimizer**: AdamW with weight decay 0.01
- **Scheduler**: CosineAnnealingLR
- **Data Augmentation**: Identical for CXR and Sputum across all models

Only model-specific parameters differ (e.g., `model_type`, `modality`, `gate_type`).

### Training Output

Each experiment creates a timestamped directory with:

```
checkpoints/{model}_{timestamp}/
├── plots/
│   ├── loss_curve.png              # Training/validation loss
│   ├── accuracy_curve.png          # Training/validation accuracy
│   ├── f1_curve.png                # Training/validation F1-score
│   ├── all_metrics.png             # Combined metrics plot
│   └── confusion_matrix_test.png   # Test set confusion matrix
├── checkpoint_best.pth             # Best model (highest val F1)
├── checkpoint_latest.pth           # Latest epoch
├── config.yaml                     # Training configuration
├── test_results.yaml               # Final test metrics
└── logs/                           # TensorBoard logs
```

### Run Multiple Experiments

Use the experiment runner for batch execution:

```bash
# List all available experiments
python3 run_experiments.py --list

# Run Tier 1 (7 baselines + MGM-TB-Former)
python3 run_experiments.py --suite tier1 --epochs 50

# Run ablation studies
python3 run_experiments.py --suite ablation_fis
python3 run_experiments.py --suite ablation_fmca

# Run specific experiment
python3 run_experiments.py --experiment mgm_tb_former

# Dry run (test without training)
python3 run_experiments.py --suite tier1 --dry_run
```

## 📊 Experimental Results

### Main Performance Comparison

We evaluated MGM-TB-Former against unimodal baselines, naive fusion methods, and state-of-the-art multimodal approaches on the JU-LDD-task-b test set.

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| CXR-Only (Unimodal) | 0.9200 | 0.9565 | 0.8800 | 0.9167 | 0.9876 | Structural baseline |
| Sputum-Only (Unimodal) | 0.7400 | 0.7308 | 0.7600 | 0.7451 | 0.7952 | Microbiological baseline |
| Concat Fusion | 0.9300 | 1.0000 | 0.8600 | 0.9247 | 0.9756 | Late fusion |
| Vanilla CMT | 0.5800 | 0.5588 | 0.7600 | 0.6441 | 0.5776 | Cross-attention (no gating) |
| Scalar Gate (MLP) | 0.6100 | 0.5902 | 0.7200 | 0.6486 | 0.6756 | Black-box gating |
| Scalar Gate (Sigmoid) | 0.5500 | 0.5510 | 0.5400 | 0.5455 | 0.5504 | Simple gating |
| ResNet-50 Fusion | 0.9600 | 0.9792 | 0.9400 | 0.9592 | 0.9964 | CNN baseline |
| EfficientNet-B0 Fusion | 0.9100 | 0.9767 | 0.8400 | 0.9032 | 0.9288 | CNN baseline |
| MobileNetV2 Fusion | 0.6300 | 0.6102 | 0.7200 | 0.6606 | 0.6760 | CNN baseline |
| **MGM-TB-Former (Ours)** | **0.9900** | **1.0000** | **0.9800** | **0.9899** | **0.9976** | **Fuzzy gating + Residual** |

### 🔑 Key Finding: The Critical Role of Residual Connections

Our ablation studies revealed a **critical architectural insight**: the residual connection is the dominant factor in MGM-TB-Former's superior performance.

| Experiment | Model | Residual? | F1-Score | Δ from MGM-TB-Former |
|:---|:---|:---:|:---:|:---:|
| Tier 1 Baseline | Vanilla CMT | ❌ No | 0.6441 | -0.3458 |
| Tier 1 Baseline | Scalar Gate (MLP) | ❌ No | 0.6486 | -0.3413 |
| **Ablation Study** | MGM-TB-Former w/o Fuzzy Gate | ✅ Yes | **0.9697** | **-0.0202** |
| **Full Model** | MGM-TB-Former | ✅ Yes | **0.9899** | — |

**What This Tells Us:**

1. **Without Residual Connection**: Cross-attention models (Vanilla CMT, Scalar Gate) perform **catastrophically** (F1 ≈ 0.55-0.65), worse than even simple concatenation (F1 = 0.9247).

2. **With Residual Connection**: Even without fuzzy gating, the model achieves F1 = 0.9697 (+32% improvement!).

3. **The Fuzzy Gate's Contribution**: The Mamdani FIS provides an additional +2% F1 improvement (0.9697 → 0.9899).

**Interpretation for the Paper:**

> *"Our ablation studies reveal that the residual connection is the primary architectural innovation enabling robust multimodal fusion on this dataset. The residual path allows the model to 'fall back' to reliable CXR features when cross-attention produces corrupted representations. The fuzzy gating mechanism provides incremental accuracy improvement (+2% F1) but, more critically, offers **interpretability** that is essential for clinical deployment."*

### 🧠 Why Fuzzy Logic Matters: Explainability

While the quantitative improvement from FIS is modest (+2% F1), the **primary contribution is interpretability**. This is crucial for clinical AI deployment:

#### Black-Box Gate (MLP/Sigmoid)

```
uncertainties → [Neural Network] → gate value (0.73)
Why 0.73? Unknown. The weights are opaque.
```

#### Fuzzy Gate (Interpretable)

```
uncertainty_cxr = 0.15 (LOW)    → "CXR is reliable"
uncertainty_sputum = 0.82 (HIGH) → "Sputum is unreliable"

Rule Fired: IF cxr_uncertainty IS LOW AND sputum_uncertainty IS HIGH 
            THEN fusion_weight IS LOW (0.25)

Interpretation: "Model is trusting CXR more because sputum quality is poor"
```

**Benefits for Clinical Deployment:**

1. **Clinical Trust**: Doctors can understand *why* the model weighted one modality over another for each patient. A model that says "TB detected" without explanation will not be trusted.

2. **Debugging & Quality Control**: If the model fails on a case, clinicians can inspect which fuzzy rules fired and whether the membership functions are calibrated correctly. This enables targeted model improvement.

3. **Regulatory Compliance**: Medical AI increasingly requires explainability (e.g., FDA guidelines, EU AI Act). Fuzzy logic provides built-in audit trails with human-readable decision rules.

4. **Alert Triggering**: When the model detects high uncertainty in a modality ($\beta \to 0$), this can automatically trigger quality control alerts (e.g., "Sputum image quality too low, please re-capture").

### The Feature Wash-out Phenomenon

We observed a critical failure mode in standard cross-attention architectures:

| Model | F1-Score | Problem |
|:---|:---:|:---|
| CXR-Only | 0.9167 | Strong unimodal baseline |
| Vanilla CMT | **0.6441** | Cross-attention *hurts* performance! |
| **MGM-TB-Former** | **0.9899** | Gated fusion *helps* |

**What Happened?**

1. Vanilla CMT has cross-attention that attends to *all* sputum features equally.
2. When sputum is noisy/unreliable, these corrupted features "wash out" the clean CXR representation.
3. The model performs *worse* than using CXR alone!

**How MGM-TB-Former Solves This:**

1. **Fuzzy Gating ($\beta$):** Detects high uncertainty in sputum → suppresses attention weights.
2. **Residual Connection:** `fused = CXR_feats + β × CrossAttn(CXR, Sputum)` ensures fallback to reliable CXR features when $\beta \to 0$.

## ⚡ Performance Notes for M4 Air

- **Batch size**: 4-8 recommended (16GB RAM)
- **Mixed precision**: Not required (MPS handles fp32 efficiently)
- **Gradient checkpointing**: Optional for larger batches
- **Expected speed**: ~30 min/epoch with 1000 training images
