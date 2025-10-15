"""
ResNet50 Confusion Matrix Visualization
=======================================

Objective:
- Visualize the confusion matrix comparing true vs predicted labels for ResNet50.
- Annotated for clarity to analyze class-wise performance.
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ===============================================================
# Settings
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

data_dir = os.path.join("dataset", "preprocessed_images")  # Relative path
model_path = os.path.join("dataset", "best_model.pth")     # Relative path

img_height, img_width = 224, 224
batch_size = 32

# ===============================================================
# Data Transform
# ===============================================================
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)
class_labels = dataset.classes

# Split into train/validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ===============================================================
# Load Model
# ===============================================================
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_labels)),
    nn.Softmax(dim=1)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ===============================================================
# Evaluation
# ===============================================================
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_labels))

# ===============================================================
# Confusion Matrix Plot
# ===============================================================
conf_matrix = confusion_matrix(all_labels, all_preds)

def plot_confusion_matrix(cm, class_labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(conf_matrix, class_labels)

