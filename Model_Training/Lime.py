"""
LIME-Based Model Interpretability for ResNet50
==============================================

Objective:
- Visualize regions of images contributing to class predictions.
- Shows explanations for 3 validation samples using LIME.
"""

import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ===============================================================
# Settings
# ===============================================================
data_dir = os.path.join("dataset", "preprocessed_images")  # Relative path
model_path = os.path.join("dataset", "best_model.pth")     # Relative path
num_samples = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ===============================================================
# Data Transform
# ===============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Dataset for LIME visualization
dataset = datasets.ImageFolder(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
_, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

# Dataset for visualization (PIL images, no tensor transform)
val_dataset_vis = datasets.ImageFolder(data_dir, transform=transforms.Compose([
    transforms.Resize((224, 224))
]))

class_labels = dataset.classes  # Automatically get class names

# ===============================================================
# Load ResNet50 Model
# ===============================================================
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(512),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(512, len(class_labels)),
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ===============================================================
# LIME Explainer
# ===============================================================
explainer = lime_image.LimeImageExplainer()

def batch_predict(images):
    """Prediction function for LIME"""
    model.eval()
    batch = torch.stack([transform(Image.fromarray(img)) for img in images], dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()

# ===============================================================
# Generate LIME Explanations
# ===============================================================
for i in range(num_samples):
    img_path, _ = val_dataset_vis.samples[i]
    image = Image.open(img_path).convert("RGB")
    np_img = np.array(image)

    explanation = explainer.explain_instance(
        np_img,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f"LIME Explanation - Predicted: {class_labels[top_label]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

