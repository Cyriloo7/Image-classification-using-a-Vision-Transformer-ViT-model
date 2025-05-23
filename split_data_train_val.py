import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Original dataset directory
source_dir = r"C:\Users\cyril\Documents\GitHub\Infolks_python_dev\Training Projects\Project 45 - skip classification\data\skip"  # e.g., "Project 45 - skip classification/skip classification"
dest_dir = r"C:\Users\cyril\Documents\GitHub\Infolks_python_dev\Training Projects\Project 45 - skip classification\data"  # e.g., "Project 45 - skip classification/data"
# Target directories
train_dir = os.path.join(dest_dir, "train")
val_dir = os.path.join(dest_dir, "val")

# Create train and val folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Parameters
val_split = 0.2
random_seed = 42

# Collect file paths and labels
file_paths = []
labels = []

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if os.path.isdir(class_path) and class_name not in ["train", "val"]:
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(class_path, filename))
                labels.append(class_name)

# Split data
train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=val_split, random_state=random_seed, stratify=labels
)

# Copy files
for filepath, label in zip(train_files, train_labels):
    dest_folder = os.path.join(train_dir, label)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy(filepath, os.path.join(dest_folder, os.path.basename(filepath)))

for filepath, label in zip(val_files, val_labels):
    dest_folder = os.path.join(val_dir, label)
    os.makedirs(dest_folder, exist_ok=True)
    shutil.copy(filepath, os.path.join(dest_folder, os.path.basename(filepath)))

print("Data successfully split into train and val folders with stratification.")
