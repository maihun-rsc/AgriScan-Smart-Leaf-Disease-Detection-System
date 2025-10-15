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
```

## Model Architecture, Training, and Evaluation

### 3. Model Architecture — ResNet50

The model uses ResNet50, a residual convolutional neural network with skip connections that mitigate vanishing gradients and enable training deeper networks efficiently.

### Key Features:
- Pre-trained on ImageNet for better initialization.
- Fully connected layer modified for the number of tomato classes.
- Optimized using Adam optimizer and CrossEntropy loss.

```python
import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)
```

---

### 4. Training Algorithm

The training loop is designed for GPU execution (if available):

```python
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Steps in the training loop:
1. Forward pass to predict output.
2. Compute loss by comparing predictions with labels.
3. Backpropagate the error.
4. Update model weights.
5. Track accuracy and validation loss.

---

### 5. Performance Evaluation

After training:
- Model accuracy, precision, recall, and F1-score are computed.
- Confusion matrices visualize class-level performance.
- Best model weights are saved to `best_model.pth`.

---

### 6. Visualization

Random samples and predictions are displayed using matplotlib for interpretability.

---

### Training Parameters

| Parameter | Description | Example |
|------------|--------------|----------|
| Batch Size | Number of samples per step | 32 |
| Epochs | Total passes over dataset | 25 |
| Learning Rate | Step size for weight updates | 0.001 |
| Optimizer | Optimization algorithm | Adam |
| Loss Function | Measures prediction error | CrossEntropyLoss |

---

### Requirements

```bash
pip install torch torchvision matplotlib numpy opencv-python tqdm
```

---

### File Structure

```
Tomato-Leaf-Disease-Detection/
│
├── model_training.py          # Main training script
├── dataset_loader.py          # Custom PyTorch dataset
├── visualize_results.py       # Visualization utilities
├── best_model.pth             # Saved trained model
├── README.md                  # Project documentation
└── requirements.txt
```

---

### CUDA Support

The training pipeline automatically detects and utilizes your GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
```

---

### Results and Metrics

- Model Accuracy: approximately 95% on validation set.
- Rapid loss reduction due to transfer learning.
- Inference speed: approximately 15ms per image on RTX 3050.

### Visualizations include:
- Training vs Validation Accuracy
- Confusion Matrix
- Example Predictions

---




