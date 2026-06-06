import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = "raw_dataset"      # original dataset folder
OUTPUT_DIR = "dataset"          # final split dataset folder

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(42)

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def split_dataset():
    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    if not source_path.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    classes = [folder for folder in source_path.iterdir() if folder.is_dir()]

    if not classes:
        raise ValueError("No class folders found inside raw_dataset.")

    for class_folder in classes:
        class_name = class_folder.name

        images = [
            img for img in class_folder.iterdir()
            if img.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        print(f"\nClass: {class_name}")
        print(f"Total: {total}")
        print(f"Train: {len(train_images)}")
        print(f"Val: {len(val_images)}")
        print(f"Test: {len(test_images)}")

        for split_name, split_images in splits.items():
            split_class_dir = output_path / split_name / class_name
            create_dir(split_class_dir)

            for image in split_images:
                shutil.copy(image, split_class_dir / image.name)

    print("\nDataset split completed successfully.")

if __name__ == "__main__":
    split_dataset()