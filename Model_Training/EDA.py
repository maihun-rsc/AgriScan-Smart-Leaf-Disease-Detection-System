"""
Exploratory Data Analysis (EDA) - Tomato Leaf Dataset
=====================================================

### Visualization Summary
1. Sample Images:
   Displays 3 random samples from each class in 'dataset/preprocessed_images'.
2. Class Distribution:
   Shows a bar plot of image counts per class to highlight any imbalances.
3. Details:
   Prints the image count for each class in the console.

=====================================================
"""

import os
import random
import cv2
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# ===============================================================
# Environment Setup
# ===============================================================

# Check for CUDA availability
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nTorch device in use: {torch_device.upper()}")

# Define dataset directory (relative for GitHub portability)
preprocessed_dir = os.path.join("dataset", "preprocessed_images")

# Configure visualization style
sns.set(style="whitegrid")

# ===============================================================
# 1. Visualize 3 Sample Images from Each Class
# ===============================================================
print("\nVisualizing 3 samples from each class...")

# Get list of classes
classes = sorted(os.listdir(preprocessed_dir))
num_classes = len(classes)

# Create subplot grid
fig, axes = plt.subplots(nrows=num_classes, ncols=3, figsize=(12, num_classes * 2.5))
if num_classes == 1:
    axes = [axes]  # Handle single-class case

for i, class_name in enumerate(classes):
    class_path = os.path.join(preprocessed_dir, class_name)
    img_files = os.listdir(class_path)
    sample_images = random.sample(img_files, min(3, len(img_files)))

    for j in range(3):
        ax = axes[i][j] if num_classes > 1 else axes[j]
        if j < len(sample_images):
            img_path = os.path.join(class_path, sample_images[j])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.axis('off')

        if j == 1:
            ax.set_title(class_name, fontsize=12, pad=10)
        ax.axis('off')

plt.tight_layout(pad=2.0)
plt.show()

# ===============================================================
# 2. Class Distribution Bar Plot
# ===============================================================
print("\nChecking class distribution...")

class_counts = {
    class_name: len(os.listdir(os.path.join(preprocessed_dir, class_name)))
    for class_name in classes
}

plt.figure(figsize=(12, 6))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
plt.xlabel("Class", fontsize=14)
plt.ylabel("Number of Images", fontsize=14)
plt.title("Class Distribution", fontsize=16)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ===============================================================
# 3. Optional: Class Distribution Pie Chart
# ===============================================================
plt.figure(figsize=(8, 8))
plt.pie(
    class_counts.values(),
    labels=class_counts.keys(),
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("pastel")
)
plt.axis("equal")
plt.title("Class Distribution Pie Chart", fontsize=16)
plt.tight_layout()
plt.show()

# ===============================================================
# 4. Class Distribution Summary
# ===============================================================
print("\nClass distribution details:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

