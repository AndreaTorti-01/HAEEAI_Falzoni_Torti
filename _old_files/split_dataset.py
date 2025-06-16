import os
import shutil
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

DATASET_DIR = 'dataset'
TESTSET_DIR = 'dataset_test'
SPLIT_RATIO = 0.2  # 20% for testing

# Create testset directory if it doesn't exist
os.makedirs(TESTSET_DIR, exist_ok=True)

# Iterate over each class folder in dataset
dataset_path = Path(DATASET_DIR)
for class_dir in dataset_path.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        files = list(class_dir.glob('*.npy'))
        if not files:
            continue
        random.shuffle(files)
        n_test = int(len(files) * SPLIT_RATIO)
        test_files = files[:n_test]

        # Create corresponding class folder in testset
        test_class_dir = Path(TESTSET_DIR) / class_name
        os.makedirs(test_class_dir, exist_ok=True)

        # Move test files to testset folder
        for f in test_files:
            shutil.move(str(f), str(test_class_dir / f.name))

print('Dataset split complete. Test files moved to', TESTSET_DIR)
