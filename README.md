<div align="center">

# MGM-TB-Net: A Mamdani Fuzzy Inference–Gated Multimodal Hybrid Transformer with Residual Fallbacks for Tuberculosis Detection

### Neuro-Fuzzy Fusion for Robust Tuberculosis Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Apple Silicon](https://img.shields.io/badge/Optimized-Apple%20M4-brightgreen?logo=apple)](https://developer.apple.com/metal/pytorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:** [Debgandhar Ghosh](mailto:debgandhar4000@gmail.com)<sup>1</sup>, Asya Lyanova<sup>2</sup>, Vyacheslav Gulvanskii<sup>2</sup>, [Dmitrii Kaplun](mailto:dikaplun@etu.ru)<sup>3,2*</sup>, [Pawan Kumar Singh](mailto:pksingh.it@jadavpuruniversity.in)<sup>1*</sup>

<sup>1</sup> Department of Information Technology, Jadavpur University, Kolkata, India
<sup>2</sup> Intelligent Devices Institute, Saint Petersburg Electrotechnical University "LETI", Russia
<sup>3</sup> Artificial Intelligence Center, Skolkovo Institute of Science and Technology, Moscow, Russia
<sup>*</sup> Corresponding authors

</div>

---

## 🔬 Overview

**MGM-TB-Net** is a novel **Mamdani Fuzzy Inference–Gated Multimodal Hybrid Transformer** architecture with **residual fallbacks** for robust Tuberculosis (TB) detection. It fuses **Chest X-Ray (CXR)** and **Sputum Smear Microscopy** images using a **Differentiable Mamdani Fuzzy Inference System (FIS)** that dynamically gates cross-modal attention based on estimated modality uncertainty, preventing the "Feature Wash-out" failure mode common in naive fusion strategies.

The proposed framework is motivated by the hierarchical nature of clinical TB screening, where CXR is used for primary screening followed by a confirmatory sputum test. MGM-TB-Net uses an **asymmetric reliability-aware** design — only the auxiliary sputum modality confidence (β) modulates cross-attention, preserving the primary CXR representation under auxiliary degradation.

> [!NOTE]
> This repository contains the official PyTorch implementation, all baseline experiments, ablation studies, and benchmark evaluations across three datasets.

---

## ⚠️ The Problem: Feature Wash-out

Standard multimodal fusion (concatenation, vanilla cross-attention) assumes all modalities are equally reliable. In clinical TB diagnosis:

- **CXR** is the primary structural modality — reliable but sensitive to sensor/positioning quality.
- **Sputum Microscopy** provides microbiological evidence — highly diagnostic but prone to staining artifacts, field-of-view bias, and focus noise.

> [!CAUTION]
> When sputum quality is degraded, naive fusion allows corrupted features to "wash out" the clean CXR representation, actively degrading accuracy below the unimodal baseline.

---

## 💡 The Solution: Neuro-Fuzzy Safety-Critical Gating

MGM-TB-Net introduces three key components:

1. **Dual-Stream Hybrid CNN–Transformer Encoder**: Modality-specific feature extraction using a multi-layer convolutional stem for local inductive bias and Transformer blocks with 2D-RoPE for global context.
2. **Uncertainty-Aware Mamdani Fuzzy Gating Module**: A differentiable Type-1 Mamdani FIS with learnable Gaussian membership functions translates entropy-based uncertainty into interpretable confidence scalars (α, β).
3. **Fuzzy-Modulated Cross-Attention (FMCA) with Residual Fallback**: The confidence scalar β gates cross-attention *post-softmax*, and a residual connection ensures graceful fallback to CXR when sputum is unreliable.

> [!TIP]
> When β → 0 (high sputum uncertainty), the fusion degrades gracefully to a unimodal CXR classifier via the residual path: `f_fused = f_CXR + FMCA`.

---

## Architecture

### Hybrid Encoder (per modality, weight-independent)

- **Convolutional Stem**: 4-layer conv stack (48 → 96 → 192 → 192 filters, stride 2), producing a 14×14×192 feature map from 224×224 input
- **Transformer Body**: 4 Transformer blocks (**192-dim**, 8 heads) with **2D Rotary Positional Embeddings (2D-RoPE)**
- Resultant sequence: **196 tokens × 192-dim**

### Differentiable Mamdani FIS

- **Inputs**: Uncertainty scalars U_cxr, U_spt (computed via normalized entropy over feature distributions)
- **Fuzzification**: 3 learnable Gaussian membership functions per input (Low / Medium / High)
- **Rule Base**: 3×3 = 9 fuzzy rules with product T-norm firing strengths
- **Implication**: Soft-min Mamdani implication
- **Aggregation**: Soft-max aggregation
- **Defuzzification**: Centroid of Area (CoA) → confidence scalars α (CXR), β (Sputum), clipped to [0, 1]

> [!IMPORTANT]
> Only β (sputum confidence) is used for cross-attention modulation. α is retained for interpretability and uncertainty reporting but is **not** used during fusion, ensuring the primary CXR representation is never suppressed.

### Fuzzy-Modulated Cross-Attention (FMCA)

```
FMCA = (Softmax(Q_CXR · K_SPT^T / √d_k) · β) · V_SPT
f_fused = f_CXR + FMCA
```

Post-softmax gating directly suppresses the magnitude of the attention update vector. Combined with the residual connection, this provides both noise suppression and safe unimodal fallback.

### Training Objective

```
L_total = L_CE(ŷ, y) + λ · L_aux(U_cxr, U_spt)
L_aux = −(Std(U_cxr) + Std(U_spt)) + γ · L_mono(μ^α, μ^β)
L_mono = Σ ReLU(μ_High − μ_Low)
```

- λ = 0.1 (variance regularization)
- γ = 0.1 (monotonicity preservation)

---

## 📊 Datasets

| # | Dataset | Task | Modalities | Source |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **Curated Label-Paired Multimodal Dataset** | TB detection (multimodal) | CXR + Sputum Microscopy | [Kaggle](https://www.kaggle.com/datasets/debg48/ju-ldd-task-b) |
| 2 | **Dataset of Tuberculosis Chest X-rays Images** | TB vs Normal (CXR-only) | CXR | [Mendeley Data](https://data.mendeley.com/datasets/8j2g3csprk/2) |
| 3 | **Tuberculosis (TB) Chest X-ray Database** | TB vs Normal (CXR-only) | CXR | [IEEE DataPort](https://ieee-dataport.org/documents/tuberculosis-tb-chest-x-ray-database) |

- **Dataset 1** is curated from two open-source datasets using a hash-based duplicate-free sampling algorithm (Algorithm 1 in paper). Split ratio: 80:10:10.
- **Datasets 2 & 3** are used for CXR-only comparative benchmarks to validate that MGM-TB-Net generalizes beyond label-paired multimodal settings.

---

## 🛠️ Installation

```bash
git clone https://github.com/debg48/MG-CMT.git
cd MG-CMT

python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Requirements:** macOS 13.0+ (MPS), Python 3.9+, 16GB RAM recommended.

---

## 📂 Project Structure

```text
.
├── models/
│   ├── encoders.py           # Custom 4-layer ViT with 2D-RoPE
│   ├── fis.py                # Differentiable Mamdani FIS
│   ├── fusion.py             # FMCA (post-softmax gating)
│   └── mgm_tb_net.py         # Full MGM-TB-Net architecture
├── baselines/
│   ├── cnn_baselines.py      # ResNet-50, EfficientNet-B0, MobileNetV2
│   └── transformer_baselines.py  # ViT, Swin Transformer, LeViT
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

## 🚀 Running Experiments

### Hyperparameters (identical across all models for fair comparison)

| Hyperparameter | Value |
| :--- | :---: |
| Image Size | 224 × 224 |
| Patch Size | 16 |
| Embedding Dim | **192** |
| Transformer Layers | 4 |
| Attention Heads | 8 |
| Dropout | 0.3 |
| Epochs | 30 |
| Batch Size | 4 |
| Learning Rate | **3e-4** |
| Weight Decay | 0.15 |
| Optimizer | AdamW |
| Scheduler | CosineAnnealingLR |
| Num Workers | 2 |

> Transformer-specific hyperparameters (Patch Size, Transformer Layers, Embedding Dim, Attention Heads) are not applicable to CNN-only baselines.

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
# Ablation 1: FIS gate variants
python3 run_experiments.py --experiment mgm_tb_net_no_gate --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_mlp_gate --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_sigmoid_gate --epochs 30

# Ablation 2: FMCA modulation strategy variants
python3 run_experiments.py --experiment fmca_standard --epochs 30
python3 run_experiments.py --experiment fmca_post_scale --epochs 30
```

### Dataset 2 & 3 Comparative Benchmarks (CXR-only)

```bash
# Dataset 2
python3 run_experiments.py --experiment densenet201_ds2 --epochs 30
python3 run_experiments.py --experiment resnet_50_ds2 --epochs 30
python3 run_experiments.py --experiment vit_ds2 --epochs 30
python3 run_experiments.py --experiment swin_ds2 --epochs 30
python3 run_experiments.py --experiment levit_ds2 --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_dataset2 --epochs 30

# Dataset 3
python3 run_experiments.py --experiment densenet201_ds3 --epochs 30
python3 run_experiments.py --experiment resnet_50_ds3 --epochs 30
python3 run_experiments.py --experiment vit_ds3 --epochs 30
python3 run_experiments.py --experiment swin_ds3 --epochs 30
python3 run_experiments.py --experiment levit_ds3 --epochs 30
python3 run_experiments.py --experiment mgm_tb_net_dataset3 --epochs 30
```

### 🔬 Robustness & Missing Modality Analysis

```bash
python3 analyze_robustness.py
```

---

## 📈 Experimental Results

### Main Comparison — Dataset 1 (Curated Multimodal)

| Model | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Unimodal CXR Model | 92.00 | 0.9565 | 0.8800 | 0.9167 | 0.9876 |
| Unimodal Sputum Model | 74.00 | 0.7308 | 0.7600 | 0.7451 | 0.7952 |
| Late Fusion (Concatenation) | 93.00 | 0.9778 | 0.8800 | 0.9263 | 0.9936 |
| Vanilla CMT | 58.00 | 0.5588 | 0.7600 | 0.6441 | 0.5776 |
| Scalar Gate (MLP) | 61.00 | 0.5902 | 0.7200 | 0.6486 | 0.6756 |
| Scalar Gate (Sigmoid) | 55.00 | 0.5510 | 0.5400 | 0.5455 | 0.5504 |
| ResNet-50 Fusion | 96.00 | 0.9792 | 0.9400 | 0.9592 | 0.9964 |
| EfficientNet-B0 Fusion | 96.00 | 0.9792 | 0.9400 | 0.9592 | 0.9952 |
| MobileNetV2 Fusion | 63.00 | 0.6102 | 0.7200 | 0.6606 | 0.6760 |
| **MGM-TB-Net (Proposed)** | **99.00** | **1.0000** | **0.9800** | **0.9899** | **0.9976** |

### Ablation Study 1 — Gating Mechanism (Dataset 1)

| Model | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| MGM-TB-Net w/ No Gate | 97.00 | 0.9796 | 0.9600 | 0.9697 | 0.9940 |
| MGM-TB-Net w/ Sigmoid Gate | 94.00 | 1.0000 | 0.8800 | 0.9362 | 0.9932 |
| MGM-TB-Net w/ MLP + Sigmoid Gate | 95.00 | 1.0000 | 0.9000 | 0.9474 | 0.9948 |
| **MGM-TB-Net (Mamdani FIS)** | **99.00** | **1.0000** | **0.9800** | **0.9899** | **0.9976** |

### Ablation Study 2 — FMCA Modulation Strategy (Dataset 1)

| Model | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FMCA (Standard, no gating) | 97.00 | 0.9610 | 0.9800 | 0.9703 | 0.9968 |
| FMCA (Logit Scaling) | 96.00 | 0.9600 | 0.9600 | 0.9600 | 0.9936 |
| **MGM-TB-Net (Post-Softmax Scaling)** | **99.00** | **1.0000** | **0.9800** | **0.9899** | **0.9976** |

### Comparative Analysis — Dataset 2 (Dataset of Tuberculosis Chest X-rays Images)

| Model | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DenseNet201 | 99.34 | 0.9947 | 0.9973 | 0.9960 | 0.9997 |
| ResNet50 | 99.78 | 0.9973 | 1.0000 | 0.9987 | 1.0000 |
| ViT | 85.84 | 0.8591 | 0.9920 | 0.9208 | 0.9329 |
| Swin Transformer | 82.96 | 0.8296 | 1.0000 | 0.9069 | 0.5841 |
| LeViT | 99.78 | 0.9973 | 1.0000 | 0.9987 | 1.0000 |
| **MGM-TB-Net (Proposed)** | **99.78** | **0.9973** | **1.0000** | **0.9987** | **1.0000** |

### Comparative Analysis — Dataset 3 (TB Chest X-ray Database)

| Model | Accuracy (%) | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DenseNet201 | 90.32 | 0.6509 | 0.9821 | 0.7829 | 0.9850 |
| ResNet50 | 97.46 | 0.9138 | 0.9464 | 0.9298 | 0.9952 |
| ViT | 95.56 | 0.9200 | 0.8214 | 0.8679 | 0.9766 |
| Swin Transformer | 82.70 | 0.5789 | 0.0982 | 0.1679 | 0.6481 |
| LeViT | 97.78 | 0.9804 | 0.8929 | 0.9346 | 0.9930 |
| **MGM-TB-Net (Proposed)** | **98.28** | **0.9391** | **0.9643** | **0.9515** | **0.9979** |

### 🔬 Robustness Analysis

**Noise Sensitivity:** Sputum confidence β decreases monotonically from 0.3935 → 0.0923 as Gaussian noise σ increases from 0.0 to 1.0, confirming dynamic down-weighting of degraded modality.

**Missing Modality Simulation:**

| Scenario | Accuracy |
| :--- | :---: |
| Both modalities present | 99.00% |
| Missing Sputum (zero tensors) | 99.00% |
| Missing CXR (zero tensors) | ~50.00% |

> The residual fallback mechanism preserves unimodal CXR functionality when sputum is unavailable. Performance degrades when CXR is missing, reflecting the real-world clinical hierarchy where CXR is the primary diagnostic modality.

---

## 📂 Key Contributions

1. A novel Mamdani fuzzy inference–gated multimodal hybrid Transformer with residual fallbacks for reliability-aware TB detection.
2. A differentiable Type-1 Mamdani FIS integrated in a DL framework for modality-specific confidence estimation with intrinsic interpretability.
3. An asymmetric FMCA mechanism where only auxiliary sputum confidence regulates cross-attention, ensuring stable CXR representation.
4. A hybrid CNN–Transformer encoder with multi-layer convolutional stem and 2D-RoPE for local inductive bias and global context.
5. Extensive experiments with 9 baselines, 2 ablation studies, robustness analysis, and evaluation on 2 additional unimodal datasets.

---

## 📦 Training Output

Each run creates a timestamped checkpoint directory:

```text
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

## ⚡ Performance Notes (Apple M4)

- Recommended batch size: 4 (for 16GB RAM)
- MPS backend handles fp32 efficiently; mixed precision not required
- All experiments performed on Apple M4 chip with 10 CPU cores and 8 GPU cores

---

## 💰 Funding

This work was supported by the **Russian Science Foundation** (Project №25-71-30008).

---

## 📝 Citation

```bibtex
@article{ghosh2026mgmtbnet,
  title={MGM-TB-Net: A Mamdani Fuzzy Inference–Gated Multimodal Hybrid Transformer
         with Residual Fallbacks for Tuberculosis Detection},
  author={Ghosh, Debgandhar and Lyanova, Asya and Gulvanskii, Vyacheslav
          and Kaplun, Dmitrii and Singh, Pawan Kumar},
  year={2026}
}
```

---

## 📧 Contact

- **Debgandhar Ghosh** — [debgandhar4000@gmail.com](mailto:debgandhar4000@gmail.com) (Jadavpur University)
- **Dmitrii Kaplun** — [dikaplun@etu.ru](mailto:dikaplun@etu.ru) (Skoltech / SPb ETU "LETI") — *Corresponding author*
- **Pawan Kumar Singh** — [pksingh.it@jadavpuruniversity.in](mailto:pksingh.it@jadavpuruniversity.in) (Jadavpur University) — *Corresponding author*
