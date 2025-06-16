import pandas as pd
import numpy as np

# --- CONFIGURATION ---
csv_path = "labels_test.csv"
output_csv = "labels_test_fft.csv"
window_len = 100         # Number of time samples per window
num_channels = 6         # IMU channels

# 1. Load the data
df = pd.read_csv(csv_path)
labels = df["label"].values
X = df.drop(columns=["label"]).values.astype(np.float32)

# 2. Compute FFT features: magnitude and phase for all coefficients
def get_fft_features(window_flat, win_len, n_channels):
    window = window_flat.reshape(win_len, n_channels)
    # For each channel, compute FFT, then stack magnitude and phase
    features = []
    for ch in range(n_channels):
        fft_vals = np.fft.fft(window[:, ch])
        mag = np.abs(fft_vals)
        phase = np.angle(fft_vals)
        features.append(mag)
        features.append(phase)
    return np.hstack(features)

X_fft = np.array([get_fft_features(row, window_len, num_channels) for row in X])

# 3. Save as new CSV
df_fft = pd.DataFrame(X_fft)
df_fft["label"] = labels
df_fft.to_csv(output_csv, index=False)
print(f"✅ Saved FFT-only features to {output_csv}")
