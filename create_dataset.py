import os
import shutil
import random

# proportion of data to use for training
TRAIN_SPLIT = 0.8

def main():
    src_root = 'snippets'
    dst_root = 'dataset'
    # find labels
    labels = [d for d in os.listdir(src_root)
              if os.path.isdir(os.path.join(src_root, d))]
    # count samples per label
    counts = {}
    for label in labels:
        files = [f for f in os.listdir(os.path.join(src_root, label))
                 if f.endswith('.npy')]
        counts[label] = len(files)
    min_count = min(counts.values())
    print("Found labels:", labels)
    print("Counts per label:", counts)
    print(f"Using {min_count} samples per label (balanced)")

    random.seed(42)
    for label in labels:
        src_dir = os.path.join(src_root, label)
        all_files = [os.path.join(src_dir, f)
                     for f in os.listdir(src_dir) if f.endswith('.npy')]
        random.shuffle(all_files)
        selected = all_files[:min_count]
        n_train = int(min_count * TRAIN_SPLIT)
        splits = {
            'train': selected[:n_train],
            'test':  selected[n_train:]
        }
        for subset, files in splits.items():
            out_dir = os.path.join(dst_root, subset, label)
            os.makedirs(out_dir, exist_ok=True)
            for fp in files:
                shutil.copy(fp, out_dir)
        print(f"{label}: {len(splits['train'])} train, {len(splits['test'])} test")

if __name__ == "__main__":
    main()
