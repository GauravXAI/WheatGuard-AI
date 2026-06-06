import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# -----------------------------
# Config
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.0001

DATA_DIR = "dataset"
MODEL_PATH = "wheat_disease_resnet18.pth"
CLASS_PATH = "class_names.json"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Set this True for Wheat Disease Detection project
USE_ONLY_WHEAT_CLASSES = False

# Based on your visible folder names
WHEAT_CLASSES = [
    "Healthy",
    "septoria",
    "stripe_rust"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Custom Dataset
# -----------------------------
class CleanImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, selected_classes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.selected_classes = selected_classes

        self.samples = []
        self.class_names = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder not found: {root_dir}")

        folders = [
            folder for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ]

        valid_classes = []

        for folder in folders:
            class_path = os.path.join(root_dir, folder)

            image_files = [
                file for file in os.listdir(class_path)
                if file.lower().endswith(IMAGE_EXTENSIONS)
            ]

            # Ignore empty folders like PlantVillage
            if len(image_files) == 0:
                print(f"Ignoring empty/non-image folder: {folder}")
                continue

            # Keep only selected wheat classes
            if selected_classes is not None and folder not in selected_classes:
                print(f"Ignoring non-wheat class: {folder}")
                continue

            valid_classes.append(folder)

        valid_classes = sorted(valid_classes)

        if len(valid_classes) == 0:
            raise ValueError(
                f"No valid image classes found in {root_dir}. "
                f"Check your dataset folders."
            )

        self.class_names = valid_classes
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.class_names)
        }

        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)

            for file in os.listdir(class_path):
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_path = os.path.join(class_path, file)
                    label = self.class_to_idx[class_name]
                    self.samples.append((image_path, label))

        print(f"\nLoaded dataset from: {root_dir}")
        print(f"Classes: {self.class_names}")
        print(f"Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------
# Transforms
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Dataset Loading
# -----------------------------
selected_classes = WHEAT_CLASSES if USE_ONLY_WHEAT_CLASSES else None

train_dataset = CleanImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transforms,
    selected_classes=selected_classes
)

val_dataset = CleanImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=eval_transforms,
    selected_classes=selected_classes
)

test_dataset = CleanImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=eval_transforms,
    selected_classes=selected_classes
)

class_names = train_dataset.class_names

if val_dataset.class_names != class_names:
    raise ValueError("Train and validation classes do not match.")

if test_dataset.class_names != class_names:
    raise ValueError("Train and test classes do not match.")

print("\nFinal Classes Used:", class_names)
print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

with open(CLASS_PATH, "w") as f:
    json.dump(class_names, f)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Model
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last ResNet block for fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, len(class_names))
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4}
])


# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

best_val_acc = 0.0


# -----------------------------
# Training
# -----------------------------
for epoch in range(EPOCHS):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("Best model saved.")


print("\nTraining completed.")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


# -----------------------------
# Testing
# -----------------------------
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.colorbar()

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()