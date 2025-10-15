# Model Training

# Tomato Leaf Disease Detection — Model Training (PyTorch + CUDA)

## Overview
This project implements a deep learning–based image classification model using PyTorch to detect and classify tomato leaf diseases. The model leverages ResNet50, a pre-trained convolutional neural network, fine-tuned on a curated dataset of healthy and diseased tomato leaf images.

The training pipeline is optimized for GPU acceleration (CUDA) and includes data preprocessing, augmentation, visualization, and performance evaluation.

---

## Objectives
- Classify tomato leaf images into healthy and diseased categories.
- Utilize transfer learning (ResNet50) to improve accuracy and reduce training time.
- Implement GPU-accelerated training with CUDA.
- Ensure model generalization through augmentation and validation.

---

## Algorithm Explanation

### 1. Data Preprocessing
Each image is resized and normalized to make it suitable for deep learning input.

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
