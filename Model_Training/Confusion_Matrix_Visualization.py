"""
Confusion Matrix Visualization for ResNet50 Predictions
=======================================================

Objective:
- Visualize the confusion matrix comparing true and predicted labels.
- Annotated for clarity and interpretability.

=======================================================
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================================================
# Settings
# ===============================================================
model_path = os.path.join("dataset", "best_model.pth")  # Relative path for GitHub
data_dir = os.path.join("dataset", "preprocessed_images")
batch_size = 32
img_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ===============================================================
# Data Transform
# ===============================================================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load validation dataset
val_dataset = datasets.ImageFolder(data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
class_names = val_dataset.classes

# ===============================================================
# Load Model
# ===============================================================
model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.5),
    nn.Linear(512, len(class_names)),
    nn.Softmax(dim=1)
)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ===============================================================
# Evaluation
# ===============================================================
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Classification Report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ===============================================================
# Confusion Matrix Visualization
# ===============================================================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.title("Confusion Matrix for ResNet50 Predictions", fontsize=15)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
