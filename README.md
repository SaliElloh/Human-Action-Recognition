# Human Action Recognition with Pose Detection 🎬🤸

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![University of Michigan](https://img.shields.io/badge/UMich-00274C?style=flat)](https://umich.edu)

> **ECE 5831 — Pattern Recognition and Neural Networks**  
> University of Michigan, Dearborn | Fall 2023

---

## Overview

This project applies deep learning for **video-based human action recognition**, comparing two state-of-the-art architectures — **ConvLSTM** and **LRCN (Long-term Recurrent Convolutional Network)** — on the UCF50 benchmark dataset across 50 action categories.

A key contribution of this work is the integration of **real-time pose estimation** (via MediaPipe) alongside action recognition, creating a richer, more interpretable representation of human motion that goes beyond frame-level classification.

---

## Results

| Model | Accuracy | Precision | Recall | Loss |
|---|---|---|---|---|
| **LRCN** | **87.70%** | **87.60%** | **86.89%** | **0.5233** |
| ConvLSTM | 83.60% | 83.60% | 83.60% | 0.6548 |

**LRCN outperformed ConvLSTM** on all metrics — showing more stable training behavior with a consistent validation gap and significantly lower susceptibility to overfitting.

![Training Results](https://github.com/user-attachments/assets/70a3e57c-319e-48fb-85b0-e8b35abc7600)

---

## Dataset

**UCF50** — a challenging benchmark for action recognition in realistic video conditions.

- 50 action categories (Basketball, Biking, Diving, PushUps, Skateboarding, etc.)
- 6,618 video clips collected from YouTube
- High intra-class variation in viewpoint, background, and lighting

Dataset: [UCF50 — University of Central Florida](https://www.crcv.ucf.edu/data/UCF50.php)

---

## Methodology

### Data Preprocessing

- Extracted 20 uniformly sampled frames per video clip
- Resized all frames to 64×64 pixels
- Normalized pixel values to [0, 1]
- Applied one-hot encoding to action labels

### Model 1 — ConvLSTM

ConvLSTM replaces standard LSTM matrix multiplications with convolution operations, allowing it to capture both **spatial and temporal patterns** within a single recurrent cell.

**Architecture:**
- ConvLSTM layers with varying filter sizes and (3, 3) kernels
- MaxPooling3D for spatial downsampling
- TimeDistributed Dropout for regularization
- Flatten + Dense output layer
- `tanh` activation with recurrent dropout

**Findings:** Strong early-stage learning but prone to overfitting after several epochs. Early Stopping was applied to prevent further divergence between training and validation accuracy.

### Model 2 — LRCN

LRCN separates spatial and temporal processing into two distinct stages — CNN layers extract per-frame spatial features, which are then passed sequentially through an LSTM for temporal reasoning.

**Architecture:**
- TimeDistributed Conv2D layers with `relu` activation
- TimeDistributed MaxPooling2D for spatial reduction
- TimeDistributed Dropout
- LSTM layer for temporal sequence modeling
- Dense output layer with softmax

**Findings:** More stable training curve, less overfitting, and better generalization — making it the stronger architecture for this task.

### Pose Detection Integration

Beyond classification, this project integrates **MediaPipe** for real-time human pose estimation, overlaying skeletal keypoints on video frames alongside the predicted action label. This approach provides a more holistic understanding of human motion and improves interpretability of model predictions.

---

## Getting Started

### Prerequisites

```bash
pip install tensorflow keras opencv-python mediapipe numpy matplotlib scikit-learn moviepy
```

### Run the notebooks

```bash
# Clone the repository
git clone https://github.com/SaliElloh/Human-Action-Recognition
cd Human-Action-Recognition
```

Then open one of the two notebooks:

| Notebook | Description |
|---|---|
| `human_action_recognition.ipynb` | ConvLSTM and LRCN model training and evaluation |
| `human_action_recognition_and_pose_detection.ipynb` | Action recognition + real-time pose overlay |

Run in **Jupyter Notebook** or **Google Colab** (recommended for GPU access).

> **Note:** Download the UCF50 dataset from the link above and place it in the root directory before running.

---

## Key Findings

- **LRCN is the superior architecture** for this task — its decoupled spatial-temporal design generalizes better than ConvLSTM's joint approach
- **Overfitting is the primary challenge** with ConvLSTM — Early Stopping helps but LRCN avoids the problem structurally
- **Pose estimation enhances interpretability** — knowing *where* the body is provides richer context than pixel-level features alone
- At 87.70% accuracy on 50 classes, LRCN approaches state-of-the-art performance for lightweight video classification models

---

## Author

**Sali El-loh**  
M.S. Artificial Intelligence | University of Michigan — Dearborn  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/salielloh12/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/SaliElloh)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:selloh@umich.edu)

---

## License

No license specified. Contact the author for usage permissions.


