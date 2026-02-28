# MGM-TB-Net: Mamdani-Gated Multimodal Transformer for Tuberculosis Detection

**Author:** Debgandhar Ghosh  
**Date:** January 2026  
**Optimized for:** Apple M4 Air (MPS Backend)

---

## Overview

**MGM-TB-Net** is a novel **Hybrid Convolutional-Transformer** architecture for robust multimodal Tuberculosis (TB) detection. It fuses **Chest X-Ray (CXR)** and **Sputum Smear Microscopy** images using a **Differentiable Mamdani Fuzzy Inference System (FIS)** that dynamically gates cross-modal attention based on estimated modality uncertainty, preventing the "Feature Wash-out" failure mode common in naive fusion strategies.

This repository contains the official PyTorch implementation, all baseline experiments, ablation studies, and benchmark evaluations across three datasets.

---

## The Problem: Feature Wash-out

Standard multimodal fusion (concatenation, vanilla cross-attention) assumes all modalities are equally reliable. In clinical TB diagnosis:

- **CXR** is the primary structural modality — reliable but sensitive to sensor/positioning quality.
- **Sputum Microscopy** provides microbiological evidence — highly diagnostic but prone to staining artifacts, field-of-view bias, and focus noise.

When sputum quality is degraded, naive fusion allows corrupted features to "wash out" the clean CXR representation, actively degrading accuracy below the unimodal baseline.

---

## The Solution: Neuro-Fuzzy Safety-Critical Gating

MGM-TB-Net introduces three innovations to solve this:

1. **Uncertainty Estimation**: Per-modality entropy is computed from the encoder's feature distribution.
2. **Mamdani FIS**: A differentiable fuzzy controller with learnable Gaussian membership functions translates entropy into interpretable confidence scalars (α, β).
3. **Gated Residual Fusion (FMCA)**: The confidence scalar β gates the cross-attention *after softmax*, and a residual connection ensures graceful fallback to CXR when sputum is unreliable:

```
fused = CXR_features + β × CrossAttention(CXR_queries, Sputum_keys, Sputum_values)
```

When β → 0 (high sputum uncertainty), the fusion degrades gracefully to a unimodal CXR classifier.

---

## Architecture

### Hybrid Encoder (per modality, weight-independent)

- **Convolutional Stem**: 4-stage conv stack with total stride 16, producing 196 tokens of dim 256 from 224×224 input
- **Transformer Body**: 4 custom transformer blocks (256-dim, 8 heads) with **2D Rotary Positional Embeddings (2D-RoPE)**
- **Global Average Pooling** → 256-dim latent vector `h`
- ~5M parameters per encoder

### Differentiable Mamdani FIS

- **Inputs**: Uncertainty scalars U_cxr, U_spt (computed via feature entropy)
- **Fuzzification**: 3 learnable Gaussian membership functions per input (Low / Medium / High)
- **Rule Base**: 3×3 = 9 fuzzy rules, initialized with logic priors (e.g., "IF Sputum_Uncertainty IS High → Trust IS Low")
- **Defuzzification**: Weighted-average Center-of-Gravity → confidence scalars α (CXR), β (Sputum)
- **Entropy Regularization Loss**: L_aux = −Var(U) prevents the uncertainty head from collapsing to a constant

### Fuzzy-Modulated Cross-Attention (FMCA)

```
FMCA(Q, K, V, β) = β · Softmax(QKᵀ / √d_k) · V
```

Post-softmax gating directly suppresses the magnitude of the attention update vector (not just flattens the distribution). Combined with the residual connection, this provides both noise suppression and safe fallback.

### Training Objective

```
L_total = L_CE(ŷ, y) + λ · L_aux(U_cxr, U_spt)
```

λ = 0.1 empirically. L_aux = −(Var(U_cxr) + Var(U_spt)).

**Total model size: ~10.5M parameters** (vs 86M for ViT-Base multimodal).

---

## Datasets

| # | Dataset | Task | Modalities |
| :---: | :--- | :--- | :--- |
| 1 | **JU-LDD-task-b** | TB detection (multimodal) | CXR + Sputum Microscopy |
| 2 | **TB Chest Radiography Database** | TB vs Normal (CXR-only) | CXR |
| 3 | **Dataset of Tuberculosis Chest X-ray Images** | TB vs Normal (CXR-only) | CXR |

- **Dataset 1** is the primary dataset for the full multimodal MGM-TB-Net evaluation.
- **Datasets 2 & 3** are used for CXR-only comparative benchmarks against standard CNN and Transformer baselines.

---

## Installation

```bash
git clone https://github.com/debg48/mgm-tb-former.git
cd mgm-tb-former

python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements:** macOS 13.0+ (MPS), Python 3.9+, 16GB RAM recommended.

---

## Project Structure

```
.
├── models/
│   ├── encoders.py           # Custom 4-layer ViT with 2D-RoPE
│   ├── fis.py                # Differentiable Mamdani FIS
│   ├── fusion.py             # FMCA (post-softmax gating)
│   └── mgm_tb_net.py         # Full MGM-TB-Net architecture
├── baselines/
│   ├── cnn_baselines.py      # ResNet-50, EfficientNet-B0, MobileNetV2, DenseNet-121, EfficientNet-V2-S
│   └── transformer_baselines.py  # ViT-Tiny, Swin-Tiny, LeViT-128s
├── data/
│   ├── dataset.py            # Dataset loaders for all 3 datasets
│   ├── JU-LDD-task-b/        # Dataset 1 (CXR + Sputum pairs)
│   ├── TB_Chest_Radiography_Database/   # Dataset 2
│   └── Dataset of Tuberculosis Chest X-rays Images/  # Dataset 3
├── configs/                  # YAML hyperparameter configs per experiment
├── utils/                    # Metrics, visualization, statistical tests
├── results/                  # Generated plots, paper draft, results
├── train.py                  # Training entry point
├── run_experiments.py        # Batch experiment runner
├── analyze_robustness.py     # Robustness & missing-modality analysis
└── generate_dataset_distribution.py   # Dataset EDA plots
```

---

## Running Experiments

### Hyperparameters (identical across all models for fair comparison)

| Hyperparameter | Value |
| :--- | :---: |
| Image Size | 224 × 224 |
| Embedding Dim | 256 |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Epochs | 30 |
| Batch Size | 4 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (wd=0.01) |
| Scheduler | CosineAnnealingLR |

Only `model_type`, `modality`, and `gate_type` differ between experiments.

### Tier 1: Core Fusion Baselines (Dataset 1 — Multimodal)

```bash
python3 run_experiments.py --experiment cxr_only --epochs 30
python3 run_experiments.py --experiment sputum_only --epochs 30
python3 run_experiments.py --experiment concat_fusion --epochs 30
python3 run_experiments.py --experiment vanilla_cmt --epochs 30
python3 run_experiments.py --experiment scalar_gate_mlp --epochs 30
python3 run_experiments.py --experiment scalar_gate_sigmoid --epochs 30
```

### Tier 2: CNN Baselines + Proposed Model (Dataset 1 — Multimodal)

```bash
python3 run_experiments.py --experiment mgm_tb_net --epochs 30
python3 run_experiments.py --experiment resnet_fusion --epochs 30
python3 run_experiments.py --experiment efficientnet_fusion --epochs 30
python3 run_experiments.py --experiment mobilenet_fusion --epochs 30
```

### Tier 3: Ablation Studies (Dataset 1)

```bash
# FIS gate variants (Why Fuzzy over MLP/Sigmoid?)
python3 run_experiments.py --experiment mgm_tb_net_no_gate --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_mlp_gate --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_sigmoid_gate --epochs 30

# FMCA attention scaling variants (Why post-softmax?)
python3 run_experiments.py --experiment fmca_standard --epochs 30
python3 run_experiments.py --experiment fmca_post_scale --epochs 30
```

### Dataset 2 Comparative Benchmark (CXR-only)

```bash
python3 run_experiments.py --experiment densenet121 --epochs 30
python3 run_experiments.py --experiment resnet_50 --epochs 30
python3 run_experiments.py --experiment efficientnet_v2_s --epochs 30
python3 run_experiments.py --experiment vit_tiny --epochs 30
python3 run_experiments.py --experiment swin_tiny --epochs 30
python3 run_experiments.py --experiment levit_tiny --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_dataset2 --epochs 30
```

### Dataset 3 Comparative Benchmark (CXR-only)

```bash
python3 run_experiments.py --experiment densenet121_ds3 --epochs 30
python3 run_experiments.py --experiment resnet_50_ds3 --epochs 30
python3 run_experiments.py --experiment efficientnet_v2_s_ds3 --epochs 30
python3 run_experiments.py --experiment vit_tiny_ds3 --epochs 30
python3 run_experiments.py --experiment swin_tiny_ds3 --epochs 30
python3 run_experiments.py --experiment levit_tiny_ds3 --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_dataset3 --epochs 30
```

### Robustness & Missing Modality Analysis

```bash
python3 analyze_robustness.py
```

Generates in `results/`:

- `failure_case_viz.png` — Noisy CXR + Clean Sputum sensor failure simulation
- `missing_modality.png` — Graceful degradation under complete modality dropout

### Dataset Distribution & Sample Visualization

```bash
python3 generate_dataset_distribution.py
```

Generates in `results/`:

- `dataset2_distribution.png`, `dataset2_samples.png`
- `dataset3_distribution.png`, `dataset3_samples.png`

### Utility Commands

```bash
# List all registered experiments
python3 run_experiments.py --list

# Dry run (validates config without training)
python3 run_experiments.py --suite tier1 --dry_run

# TensorBoard monitoring
tensorboard --logdir checkpoints
```

---

## Training Output

Each run creates a timestamped checkpoint directory:

```
checkpoints/{model}_{timestamp}/
├── plots/
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── f1_curve.png
│   ├── all_metrics.png
│   └── confusion_matrix_test.png
├── checkpoint_best.pth
├── checkpoint_latest.pth
├── config.yaml
├── test_results.yaml
└── logs/                    # TensorBoard logs
```

---

## Experimental Results

### Main Comparison — Dataset 1 (JU-LDD-task-b, Multimodal)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| CXR-Only | 0.9200 | 0.9565 | 0.8800 | 0.9167 | 0.9876 | Unimodal baseline |
| Sputum-Only | 0.7400 | 0.7308 | 0.7600 | 0.7451 | 0.7952 | Unimodal baseline |
| Concat Fusion | 0.9300 | 1.0000 | 0.8600 | 0.9247 | 0.9756 | Naive late fusion |
| Vanilla CMT | 0.5800 | 0.5588 | 0.7600 | 0.6441 | 0.5776 | Cross-attention, no gating |
| Scalar Gate (MLP) | 0.6100 | 0.5902 | 0.7200 | 0.6486 | 0.6756 | Black-box gating |
| Scalar Gate (Sigmoid) | 0.5500 | 0.5510 | 0.5400 | 0.5455 | 0.5504 | Simple gating |
| ResNet-50 Fusion | 0.9600 | 0.9792 | 0.9400 | 0.9592 | 0.9964 | CNN multimodal |
| EfficientNet-B0 Fusion | 0.9100 | 0.9767 | 0.8400 | 0.9032 | 0.9288 | CNN multimodal |
| MobileNetV2 Fusion | 0.6300 | 0.6102 | 0.7200 | 0.6606 | 0.6760 | CNN multimodal |
| **MGM-TB-Net (Ours)** | **0.9900** | **1.0000** | **0.9800** | **0.9899** | **0.9976** | Fuzzy gating + Residual |

### Ablation — Critical Role of Residual Connection

| Configuration | Residual | F1-Score | Δ vs Full Model |
| :--- | :---: | :---: | :---: |
| Vanilla CMT | ❌ | 0.6441 | −0.3458 |
| MGM-TB-Net w/o Fuzzy Gate | ✅ | 0.9697 | −0.0202 |
| **MGM-TB-Net (Full)** | ✅ | **0.9899** | — |

The residual connection accounts for ~32% of the F1 gain. The fuzzy gate adds a further +2% with the critical benefit of **clinical interpretability**.

### Robustness

| Failure Scenario | Accuracy |
| :--- | :---: |
| Missing Sputum (zeroed) | 99.00% |
| Missing CXR (zeroed) | ~50% (random guess) |
| Gaussian Noise on CXR (σ=0.1–1.0) | Stable (α_cxr ≈ 0.53 across all σ) |

Noise stability indicates **feature invariance** in the convolutional stem — the encoder filters high-frequency noise before the transformer body.

### Dataset 2 & 3 Comparative Benchmarks (CXR-only)

Results to be populated after completing 30-epoch training runs for each model.

---

## Model Size Comparison

| Model | Parameters | FLOPs | Notes |
| :--- | ---: | ---: | :--- |
| ViT-Base | 86M | 17.6G | Standard pretrained |
| ViT-Small | 22M | 4.6G | |
| **MGM-TB-Net** | **10.5M** | **2.1G** | Custom lightweight |

---

## Performance Notes (Apple M4 Air)

- Recommended batch size: 4–8 (for 16GB RAM)
- Expected speed: ~30 min/epoch on ~1000 training images
- MPS backend handles fp32 efficiently; mixed precision not required

---

## Citation

```bibtex
@article{ghosh2026mgmtbnet,
  title={MGM-TB-Net: Mamdani-Gated Multimodal Transformer for Robust Tuberculosis Detection},
  author={Ghosh, Debgandhar},
  year={2026}
}
```

---

## Contact

For questions or collaborations: [debgandhar4000@gmail.com]
