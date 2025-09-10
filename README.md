# Injury Prediction for Social Compatibility

**Code & Data for “Learning Socially Compatible Autonomous Driving under Safety-Critical Scenarios”**

This repository implements **occupant injury severity prediction** using a **teacher–student knowledge distillation (KD) framework**.  
It includes:  
- A **high-fidelity TCN teacher model** (crash-pulse → injury).  
- A **student model (MLP)** trained with and without KD (initial conditions → injury).  
- Unified training, evaluation, and visualization pipelines.  

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
Accurate **injury prediction** (continuous **Head Injury Criterion – HIC**, categorical **AIS** levels) is essential for socially compatible autonomous vehicle(AV) decision-making (e.g., minimizing unfair risk allocation under unavoidable collisions).  
This project focuses on predicting **HIC (Head Injury Criterion)** and categorical **AIS levels** (6-class and 3-class) for safety-critical scenarios, building an injury surrogate that can be coupled with higher-level planning / policy learning to evaluate safety–fairness trade-offs.

Three complementary tasks are supported:

| Task | Input | Output | Model | Use Case                                               |
|------|-------|--------|-------|--------------------------------------------------------|
| **Teacher (Crash Pulse → Severity)** | Time-series crash acceleration + attributes | HIC + AIS | TCN | High-fidelity injury modeling                          |
| **Student w/ KD** | Initial collision conditions | HIC + AIS | MLP (distilled) | Fast inference and high accuracy with teacher guidance |
| **Student wo/ KD** | Initial collision conditions | HIC + AIS | MLP | Fast inference without teacher                         |

---

## Repository Structure
```
Learning-Socially-Compatible-Autonomous-Driving/
├─Codes_ART_injury_prediction/
  ├─ data/                          # Dataset
  │  ├─ data_crashpulse.npy         # (N, 2, 150) crash pulses
  │  ├─ data_features.npy           # (N, 9) features (8 conditions + HIC)
  │  ├─ train_dataset.pt
  │  ├─ val_dataset.pt
  │  └─ test_dataset.pt
  ├─ ckpt/                          # Saved checkpoints
  │  ├─ student_w_KD_best.pth
  │  ├─ student_wo_KD_best
  │  ├─ student_wo_KD_best.pth
  │  ├─ teacher_best
  │  └─ teacher_best.pth
  ├─ results/                       # Validation results (Markdown + plots)
  │  ├─ teacher_model_results.md
  │  ├─ student_model_w_KD_results.md
  │  ├─ student_model_wo_KD_results.md
  │  ├─ teacher_model.png
  │  ├─ student_model_w_KD.png
  │  └─ student_model_wo_KD.png
  ├─ utils/                         # Core modules
  │  ├─ combined_loss.py            # Weighted hybrid loss
  │  ├─ dataset_prepare.py          # Dataset & AIS label generation
  │  └─ models.py                   # Teacher & Student models (TCN / MLP)
  ├─ train_teacher.py               # Train teacher (TCN)
  ├─ train_student_w_KD.py          # Train student with KD
  ├─ train_student_wo_KD.py         # Train student without KD
  ├─ test_teacher.py                # Evaluate teacher
  ├─ test_student_w_KD.py           # Evaluate student w/ KD
  ├─ test_student_wo_KD.py          # Evaluate student wo/ KD
  └─ requirements.txt
```

---

## Key Features
- **Teacher–Student Distillation** with embedding and deep feature mimicry.  
- **Temporal Convolutional Network** with dilated causal convolutions, residual blocks, and feature fusion.
- **Knowledge Distillation**: Intermediate embedding + deep representation alignment (feature-level MSE terms).
- **Class imbalance handling**: Metrics include *G-mean* and confusion matrices.
- **Unified evaluation**: results stored as Markdown reports + confusion matrices + scatter plots.  

---

## Installation
Create and activate a dedicated environment:
```bash
conda create -n injury_pred python=3.10
conda activate injury_pred
pip install -r requirements.txt
```
Ensure you have a working PyTorch environment

---

## Dataset
- Stored in `data/` folder.  
- Includes:
  - `data_crashpulse.npy` (crash pulses for teacher model).  
  - `data_features.npy` (initial collision conditions & HIC).  
  - Pre-split datasets (`train_dataset.pt`, `val_dataset.pt`, `test_dataset.pt`).  
- AIS labels are generated automatically from HIC via logistic mapping.  

| File | Shape | Description |
|------|-------|-------------|
| `data_crashpulse.npy` | (5777, 2, 150) | Bidirectional (e.g., x / y) acceleration pulses (150 time steps each). |
| `data_features.npy` | (5777, 9) | 8 initial collision descriptors (impact kinematics / geometry etc.) + final column: HIC label. |

---

## Model Pipelines
### Teacher Model (`train_teacher.py`)
- **Architecture:** Temporal Convolutional Network (TCN).  
- **Inputs:**`(acc_x, acc_y)` sequences + categorical / scalar crash attributes
- **Outputs:** HIC (regression) + AIS classification.

### Student Model w/ KD (`train_student_w_KD.py`)
- **Architecture:** Lightweight MLP.  
- **Distillation:**  
  - HIC supervised loss.  
  - Feature-level mimicry from teacher embeddings.  
- **Outputs:** HIC + AIS with improved performance.  

### Student Model wo/ KD (`train_student_wo_KD.py`)
- **Architecture:** Lightweight MLP.  
- **No distillation signals** — trained only on ground truth.  

---

## Scripts
| Script | Purpose                                                      |
|--------|--------------------------------------------------------------|
| `train_teacher.py` | Train the teacher model (TCN) on crash pulses and attributes. |
| `train_student_w_KD.py` | Train the student model with KD using teacher guidance.      |
| `train_student_wo_KD.py` | Train the student model without knowledge distillation.      |
| `test_teacher.py` | Evaluate teacher model performance and generate results.     |
| `test_student_w_KD.py` | Evaluate student model w/ KD performance.                    |
| `test_student_wo_KD.py` | Evaluate student model wo/ KD performance.                   |

---

## Metric Definitions
| Metric | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| **RMSE / MAE** | Standard regression error metrics for HIC.                                  |
| **R²** | Coefficient of determination for HIC regression.                            |
| **Accuracy** | Correct AIS predictions / total.                                            |
| **G-mean** | $\left( \prod_{c=1}^{C} Sensitivity_c \right)^{1/C}$ — robust to imbalance. |
| **Confusion Matrix** | Distribution of predicted vs. actual AIS classes.                           |

---



## Contact
**Primary contacts:**  
- Shuo Feng (fshuo@tsinghua.edu.cn)  
- Bingbing Nie (nbb@tsinghua.edu.cn)  

**Maintainers:**  
- Jiajie Shen  
- Gaoyuan Kuang  

*Last updated: 2025-09*
