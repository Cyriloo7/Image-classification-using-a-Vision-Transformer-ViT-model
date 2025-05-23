import os
import shutil
import random
import argparse
from sklearn.model_selection import train_test_split


def split_data(source_dir, dest_dir, val_split=0.2, seed=42):
    """
    Splits dataset into training and validation sets with stratification.
    Assumes source_dir contains class subfolders (excluding 'train' and 'val').
    """
    train_dir = os.path.join(dest_dir, "train")
    val_dir = os.path.join(dest_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Gather file paths and labels
    file_paths = []
    labels = []
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if os.path.isdir(class_path) and class_name not in ["train", "val"]:
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_paths.append(os.path.join(class_path, fname))
                    labels.append(class_name)

    # Split
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=val_split, random_state=seed, stratify=labels
    )

    # Copy
    for filepath, label in zip(train_files, train_labels):
        d = os.path.join(train_dir, label)
        os.makedirs(d, exist_ok=True)
        shutil.copy(filepath, os.path.join(d, os.path.basename(filepath)))
    for filepath, label in zip(val_files, val_labels):
        d = os.path.join(val_dir, label)
        os.makedirs(d, exist_ok=True)
        shutil.copy(filepath, os.path.join(d, os.path.basename(filepath)))

    print(f"Split done. {len(train_files)} training and {len(val_files)} validation images.")


def main():
    parser = argparse.ArgumentParser(description="Train a skip classification model with ViT architecture.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the data directory containing class subfolders and skip folder')
    parser.add_argument('--val_split', type=float, default=0.2, help= 'Fraction of data to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Paths
    source_dir = os.path.abspath(args.data_dir)
    dest_dir = os.path.dirname(source_dir)

    # Split data
    split_data(source_dir, dest_dir, args.val_split, args.seed)


if __name__ == '__main__':
    main()
