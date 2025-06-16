import os
import numpy as np
import pandas as pd

snippets_dir = "dataset_test"
samples = []
labels = []

for label in os.listdir(snippets_dir):
    label_folder = os.path.join(snippets_dir, label)
    if not os.path.isdir(label_folder):
        continue
    for fname in os.listdir(label_folder):
        if not fname.endswith(".npy"):
            continue
        fpath = os.path.join(label_folder, fname)
        data = np.load(fpath)              # shape: (window_len, n_features)
        sample = data.flatten()            # shape: (window_len * n_features,)
        samples.append(sample)
        labels.append(label)

X = np.vstack(samples)
df = pd.DataFrame(X)
df['label'] = labels
df.to_csv("labels_test.csv", index=False)
print("✅ Dataset saved")
