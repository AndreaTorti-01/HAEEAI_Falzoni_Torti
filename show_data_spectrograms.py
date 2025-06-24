import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.signal import spectrogram

# File paths
imu_file = 'football_data/imu_data_20250529_183017.txt'
events_file = 'football_data/events_183017.txt'

# Read IMU data
imu_data = []
with open(imu_file, 'r') as f:
    for line in f:
        # allow spaces around the hyphen
        parts = re.split(r'\s*-\s*', line.strip(), maxsplit=1)
        if len(parts) == 2:
            try:
                timestamp = int(parts[0].strip())
                values = list(map(float, parts[1].strip().split()))
                if len(values) == 6:
                    imu_data.append([timestamp] + values)
            except ValueError:
                continue

imu_df = pd.DataFrame(imu_data, columns=['Timestamp', 'accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz'])

# Calculate magnitude of A and G
imu_df['mag_A'] = np.sqrt(imu_df['accx']**2 + imu_df['accy']**2 + imu_df['accz']**2)
imu_df['mag_G'] = np.sqrt(imu_df['gyrox']**2 + imu_df['gyroy']**2 + imu_df['gyroz']**2)

# Convert timestamps to datetime objects for plotting
imu_df['Timestamp_dt'] = pd.to_datetime(imu_df['Timestamp'], unit='ms')

# Read event data
event_data = []
event_types_of_interest = [
    'INIZIO_CAMMINATA', 'FINE_CAMMINATA', 'PASSAGGIO', 'TIRO',
    'INIZIO_CORSA', 'FINE_CORSA', 'INIZIO_STANDING', 'FINE_STANDING'
]

with open(events_file, 'r') as f:
    for line in f:
        match = re.match(r'(\d+) - (.*)', line.strip())
        if match:
            try:
                timestamp = int(match.group(1))
                action = match.group(2).strip()
                event_data.append([timestamp, action])
            except ValueError:
                continue

event_df = pd.DataFrame(event_data, columns=['Timestamp', 'Action'])

# Filter for relevant events
event_df_filtered = event_df[event_df['Action'].isin(event_types_of_interest)].copy()
event_df_filtered['Timestamp_dt'] = pd.to_datetime(event_df_filtered['Timestamp'], unit='ms')

# Compute sampling rate from timestamps (Hz)
timestamps = imu_df['Timestamp_dt'].astype(np.int64) / 1e9
dt = np.median(np.diff(timestamps))
fs = 1 / dt

# Create the plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 10))

# Plot spectrogram of magnitude A with sliding window=100 samples, overlap=90 (step=10)
f_A, t_A, Sxx_A = spectrogram(
    imu_df['mag_A'].values, fs=fs,
    window='hann', nperseg=100, noverlap=90, mode='magnitude'
)
# Convert to dB for better contrast
Sxx_A_dB = 10 * np.log10(Sxx_A + 1e-12)
vmax_A = Sxx_A_dB.max()
vmin_A = vmax_A - 60  # 60 dB dynamic range
t_A_dt = imu_df['Timestamp_dt'].iloc[0] + pd.to_timedelta(t_A, unit='s')
ax1.pcolormesh(t_A_dt, f_A, Sxx_A_dB, shading='gouraud', cmap='magma', vmin=vmin_A, vmax=vmax_A)
ax1.set_ylabel('Frequency [Hz]')
ax1.set_yscale('log')
ax1.set_title('Spectrogram of Magnitude A')
ax1.grid(True)

# Plot spectrogram of magnitude G with the same parameters
f_G, t_G, Sxx_G = spectrogram(
    imu_df['mag_G'].values, fs=fs,
    window='hann', nperseg=100, noverlap=90, mode='magnitude'
)
# Convert to dB for better contrast
Sxx_G_dB = 10 * np.log10(Sxx_G + 1e-12)
vmax_G = Sxx_G_dB.max()
vmin_G = vmax_G - 60  # 60 dB dynamic range
t_G_dt = imu_df['Timestamp_dt'].iloc[0] + pd.to_timedelta(t_G, unit='s')
ax2.pcolormesh(t_G_dt, f_G, Sxx_G_dB, shading='gouraud', cmap='magma', vmin=vmin_G, vmax=vmax_G)
ax2.set_ylabel('Frequency [Hz]')
ax2.set_yscale('log')
ax2.grid(True)

# Plot events as dashed lines with shaded intervals
colors = plt.get_cmap('Dark2', len(event_types_of_interest) // 2 + 1)
# Map event base names to a color
event_base_names = [e.split('_', 1)[1] for e in event_types_of_interest if e.startswith('INIZIO_')]
event_base_color_map = {name: colors(i) for i, name in enumerate(event_base_names)}

# Shade between INIZIO and FINE
for name in event_base_names:
    start_key = f'INIZIO_{name}'
    end_key   = f'FINE_{name}'
    starts = event_df_filtered.loc[event_df_filtered['Action'] == start_key, 'Timestamp_dt']
    ends   = event_df_filtered.loc[event_df_filtered['Action'] == end_key,   'Timestamp_dt']
    for s, e in zip(starts, ends):
        ax3.axvspan(s, e, color=event_base_color_map[name], alpha=0.15)
        # Add label in the middle of the shaded region
        mid = s + (e - s) / 2
        ax3.text(mid, 0.5, name, color=event_base_color_map[name], fontsize=10, ha='center', va='center',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.2'),
                 clip_on=True, transform=ax3.get_xaxis_transform())

# Draw vertical dashed lines for all events and add text for each event
for _, row in event_df_filtered.iterrows():
    t = row['Timestamp_dt']
    action = row['Action']
    base = action.split('_', 1)[1] if '_' in action else action
    color = event_base_color_map.get(base, 'black')
    ax3.axvline(x=t, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
    # Place text slightly above the bottom, rotated, and clipped to axes
    ax3.text(t, 0.02, action, rotation=90, ha='left', va='bottom',
             color=color, fontsize=8, backgroundcolor='white',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, boxstyle='round,pad=0.1'),
             clip_on=True, transform=ax3.get_xaxis_transform())

ax3.set_xlabel('Time')
ax3.set_ylim(0, 1)
ax3.set_yticks([])
ax3.grid(True)

plt.tight_layout()
plt.show()