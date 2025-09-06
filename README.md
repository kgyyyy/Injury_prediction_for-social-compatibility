# Injury Prediction for Social Compatibility
**Code & Data for “Learning Socially Compatible Autonomous Driving under Safety-Critical Scenarios”**

> This repository hosts the injury (HIC / AIS) prediction modules used in our study on socially compatible autonomous driving under safety-critical scenarios.  

> Provides (i) a high-capacity **Temporal Convolutional Network (TCN)** teacher model (crash-pulse → injury) and (ii) a lightweight **Student (S2S) MLP** distilled from the teacher (initial collision conditions → injury), enabling an accuracy–efficiency trade-off suitable for deployment.

---
## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Key Features](#key-features)  
4. [Installation](#installation)  
5. [Dataset](#dataset)  
6. [Model Pipelines](#model-pipelines)  
7. [Scripts](#scripts)
8. [Metric Definitions](#metric-definitions)
9. [Contact](#contact)  


---

## Project Overview
Accurate *occupant injury severity* prediction (continuous **Head Injury Criterion – HIC**, categorical **AIS** levels) is essential for socially compatible autonomous vehicle (AV) decision-making (e.g., minimizing unfair risk allocation under unavoidable collisions).  
This module builds an *injury surrogate* that can be coupled with higher-level planning / policy learning to evaluate safety–fairness trade-offs.

Two complementary tasks are supported:

| Task | Input | Output | Model | Use Case |
|------|-------|--------|-------|----------|
| **Crash-Pulse → Severity (C2S)** | Time–series acceleration (2 axes × 150) + scenario attributes | HIC (regression) + AIS (6-class & 3-class) | TCN (teacher) | High-fidelity analysis / knowledge source |
| **Initial Conditions → Severity (KD / S2S)** | 8 scalar/categorical initial collision features | HIC + AIS | Distilled MLP (student) | Fast inference, deployment, large-scale simulation |
---

## Repository Structure
```
Learning-Socially-Compatible-Autonomous-Driving/
└─ Codes_ART_injury_prediction/
   ├─ data/
   │  ├─ data_crashpulse.npy        # (N, 2, 150) acceleration pulses
   │  └─ data_features.npy          # (N, 9) collision features + HIC
   ├─ image/                        # Sample prediction / scatter plots
   ├─ params/
   │  └─ Best/                      # Saved best model weights
   ├─ utils/
   │  ├─ backbones.py               # TCN + S2S architectures & modules
   │  └─ load_data.py               # Loading, normalization, label gen
   ├─ requirements.txt
   ├─ main_C2S.py                   # Train/eval TCN (teacher)
   └─ main_KD.py                    # Train/eval distilled MLP (student)
```
---

## Key Features
- **Unified multi-task objective**: Simultaneous HIC regression + AIS classification (6- and 3-level schemes).
- **Temporal Convolutional Network** with dilated causal convolutions, residual blocks, and feature fusion.
- **Knowledge Distillation**: Intermediate embedding + deep representation alignment (feature-level MSE terms).
- **Reproducible preprocessing**: Deterministic seeding and structured train/val/test splits.
- **Computational profiling**: FLOPs / parameter counts via `thop` + per-sample latency measurement.
- **Class imbalance handling**: Metrics include *G-mean* and confusion matrices.



---
## Installation
Create and activate a dedicated environment:
```bash
conda create -n InjPred python=3.7
conda activate InjPred
pip install -r requirements.txt
```
Ensure you have a working PyTorch environment

---

## Dataset
### Files
| File | Shape | Description |
|------|-------|-------------|
| `data_crashpulse.npy` | (5777, 2, 150) | Bidirectional (e.g., x / y) acceleration pulses (150 time steps each). |
| `data_features.npy` | (5777, 9) | 8 initial collision descriptors (impact kinematics / geometry etc.) + final column: HIC label. |

### Internal Label Generation
`load_data.py` computes:
- **HIC regression target** (from final column of `data_features.npy`).
- **AIS (6-class)** & **AIS (3-class)** via biomechanically informed logistic mappings (`AIS_cal`, `AIS_3_cal`).

### Splits
Data are shuffled and partitioned into train / validation / test inside `load_data()` (see code comments for ratios).

---

## Model Pipelines
### 1. `main_C2S.py` – TCN Teacher
**Inputs:** `(acc_x, acc_y)` sequences + categorical / scalar crash attributes  
**Core Components:**
- Embedding of each acceleration channel.
- TemporalConvNet (stack of `TemporalBlock`s with dilation).
- Feature fusion with embedded attributes.
- Multi-head output: HIC (regression) + AIS logits.

**Training Loop:** Grid search over dropout, batch size, embedding size, hidden size, kernel size, depth, learning rate. Early stopping on validation accuracy / loss. Best weights saved in `params/Best/C2S_best`.

**Evaluation:**
- Regression: RMSE, MAE, R².
- Classification: Accuracy, G-mean, confusion matrices (6-class & 3-class).
- Visualization: Predicted vs. actual HIC scatter plot.
- Efficiency: Per-sample latency (batch size = 1), FLOPs, parameter count.

### 2. `main_KD.py` – Distilled Student (S2S)
**Inputs:** Initial collision condition vector only (`x_att`).  
**Distillation Signals:**
- HIC supervised loss.
- Embedding feature mimic (teacher intermediate).
- Deep feature mimic (teacher high-level representation).  
Weighted by hyperparameters `ratio_E`, `ratio_D`.

**Outcomes:** Comparable AIS / HIC performance with reduced latency and footprint.

---

## Scripts
| Script | Purpose                                                                                                                                                                                                                                                                        |
|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main_C2S.py` | Train & evaluate teacher TCN on crash pulses: The first part is Model Training, the second part is Performance Evaluation, and the third part is Model Inference Efficiency Evaluation. When executing the code for model evaluation, the model training parts need to be commented out. |
| `main_KD.py` | Train & evaluate student MLP with feature distillation: The first part is Model Training, the second part is Performance Evaluation, and the third part is Model Inference Efficiency Evaluation. When executing the code for model evaluation, the model training parts need to be commented out.                                                                                                                                                                                                                       |
| `utils/backbones.py` | Defines `TemporalBlock`, `TemporalConvNet`, `MLP`, `PositionalEncoding`, `TCN`, `S2S`.                                                                                                                                                                                         |
| `utils/load_data.py` | Loads data, normalizes features, generates AIS labels, returns splits as PyTorch tensors.                                                                                                                                                                                      |

---

## Metric Definitions
| Metric | Description |
|--------|-------------|
| **RMSE / MAE** | Standard regression error metrics for HIC. |
| **R²** | Coefficient of determination for HIC regression. |
| **Accuracy** | Correct AIS predictions / total. |
| **G-mean** | $ \left( \prod_{c=1}^{C} \mathrm{Sensitivity}_c \right)^{1/C} $ — robust to imbalance |
| **FLOPs** | Multiply–add operations estimated via `thop.profile`. |
| **Parameters** | Trainable parameter count. |
| **Inference Time** | Mean ± std per sample at `batch_size=1`. |




---

## Contact
**Primary contact:** *[Bingbing Nie/ nbb@tsinghua.edu.cn]*  
**Project Maintainers:** Jiajie Shen, Gaoyuan Kuang

---


*Last updated: 2025-09-06*
