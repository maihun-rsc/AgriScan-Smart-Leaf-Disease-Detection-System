import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# ===============================================================
# Tomato Leaf Dataset Preprocessing Script
# ===============================================================
# This script resizes and normalizes all images from the dataset
# and saves them into a new 'preprocessed_images/' directory.
#
# Expected structure before running:
#   dataset/
#       ├── train/
#       │   ├── class_1/
#       │   ├── class_2/
#       │   └── ...
#       └── valid/
#           ├── class_1/
#           ├── class_2/
#           └── ...
# ===============================================================

# === Settings ===
train_dir = os.path.join("dataset", "train")
valid_dir = os.path.join("dataset", "valid")
preprocessed_dir = os.path.join("dataset", "preprocessed_images")
target_size = (224, 224)


# === Helper Function to Load Images and Labels ===
def load_images_from_folder(folder):
    """Load all images and their corresponding class labels from the dataset folder."""
    images, labels = [], []
    print(f"\nLoading from: {folder}")

    for class_name in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue

        img_files = glob(os.path.join(class_path, "*.*"))
        for img_path in img_files:
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
                labels.append(class_name)

        print(f"{class_name}: {len(img_files)} images loaded")

    return images, labels


# === Load Images from Train and Validation Folders ===
all_images, all_labels = [], []
for subdir in [train_dir, valid_dir]:
    imgs, lbls = load_images_from_folder(subdir)
    all_images.extend(imgs)
    all_labels.extend(lbls)


# === Preprocessing and Saving ===
os.makedirs(preprocessed_dir, exist_ok=True)
processed_count = 0
print("\nStarting preprocessing and saving...\n")

for i, (img, label) in enumerate(tqdm(zip(all_images, all_labels), total=len(all_images))):
    resized_img = cv2.resize(img, target_size)
    normalized_img = resized_img / 255.0
    save_img = (normalized_img * 255).astype(np.uint8)

    class_dir = os.path.join(preprocessed_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    img_path = os.path.join(class_dir, f"{i}.jpg")
    cv2.imwrite(img_path, save_img)
    processed_count += 1

# === Summary ===
print("\nPreprocessing complete.")
print(f"Preprocessed images saved to: {preprocessed_dir}")
print(f"Total images processed: {processed_count}")
print(f"Total classes found: {len(set(all_labels))}")

