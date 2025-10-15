import os
from glob import glob
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt


class TomatoLeafDataset(Dataset):
    """
    Custom PyTorch Dataset for Tomato Leaf Classification.
    Expects the following folder structure:

        dataset/
            ├── train/
            │   ├── class_1/
            │   ├── class_2/
            │   └── ...
            └── valid/
                ├── class_1/
                ├── class_2/
                └── ...

    Each class folder should contain images belonging to that class.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset images and class labels from the specified root directory."""
        print(f"\nLoading images from: {self.root_dir}\n")
        class_names = sorted(os.listdir(self.root_dir))
        class_counts = defaultdict(int)

        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            img_files = glob(os.path.join(class_path, '*.*'))
            for img_path in img_files:
                self.image_paths.append(img_path)
                self.labels.append(idx)
                class_counts[class_name] += 1

            print(f"{class_name}: {class_counts[class_name]} images")

        print(f"\nTotal images: {len(self.image_paths)}")
        print(f"Classes found: {len(self.class_to_idx)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -------------------------------------------
# Dataset usage example
# -------------------------------------------

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Replace these with your local dataset paths
# Example directory structure:
# ├── dataset/
# │   ├── train/
# │   └── valid/
train_path = os.path.join("dataset", "train")
valid_path = os.path.join("dataset", "valid")

# Load datasets
train_dataset = TomatoLeafDataset(train_path, transform=transform)
valid_dataset = TomatoLeafDataset(valid_path, transform=transform)


def show_samples(dataset, num_samples=6):
    """
    Displays sample images from the dataset with corresponding class labels.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5)
        axes[i].imshow(img)
        axes[i].set_title(dataset.idx_to_class[label])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


# Display sample images from the training dataset
show_samples(train_dataset)

