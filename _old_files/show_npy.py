import numpy as np
import matplotlib.pyplot as plt
import os

folder = 'snippets/RUNNING'
npy_files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])

for npy_file in npy_files:
    npy_path = os.path.join(folder, npy_file)
    snippet = np.load(npy_path)
    print(f"{npy_path}: {snippet.shape}")
    plt.figure(figsize=(10, 5))

    # compute and plot magnitudes
    mag_A = np.linalg.norm(snippet[:, :3], axis=1)
    mag_G = np.linalg.norm(snippet[:, 3:], axis=1)
    plt.plot(mag_A, label='Mag A')
    plt.plot(mag_G, label='Mag G')

    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.title(f'Snippet {npy_file} ({snippet.shape[0]} samples)')
    plt.legend()
    plt.tight_layout()
    plt.show()
