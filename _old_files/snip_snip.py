import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── 1. LOAD & PREPARE YOUR DATA ─────────────────────────────────────────────

imu_file    = 'football_data/imu_data_20250529_183017.txt'
events_file = 'football_data/events_183017.txt'

imu_rows = []
with open(imu_file, 'r') as f:
    for line in f:
        parts = re.split(r'\s*-\s*', line.strip(), maxsplit=1)
        if len(parts) == 2:
            try:
                ts   = int(parts[0].strip())
                vals = list(map(float, parts[1].strip().split()))
                if len(vals) == 6:
                    imu_rows.append([ts] + vals)
            except ValueError:
                continue

imu_df = pd.DataFrame(
    imu_rows,
    columns=['Timestamp','accx','accy','accz','gyrox','gyroy','gyroz']
)
imu_df['mag_A'] = np.sqrt(
    imu_df['accx']**2 + imu_df['accy']**2 + imu_df['accz']**2
)
imu_df['mag_G'] = np.sqrt(
    imu_df['gyrox']**2 + imu_df['gyroy']**2 + imu_df['gyroz']**2
)
imu_df['Timestamp_dt'] = pd.to_datetime(imu_df['Timestamp'], unit='ms')

event_data = []
event_types_of_interest = [
    'INIZIO_CAMMINATA','FINE_CAMMINATA','PASSAGGIO','TIRO',
    'INIZIO_CORSA','FINE_CORSA','INIZIO_STANDING','FINE_STANDING'
]
with open(events_file, 'r') as f:
    for line in f:
        match = re.match(r'(\d+) - (.*)', line.strip())
        if match:
            try:
                ts     = int(match.group(1))
                action = match.group(2).strip()
                event_data.append([ts, action])
            except ValueError:
                continue

event_df = pd.DataFrame(event_data, columns=['Timestamp','Action'])
event_df_filtered = event_df[event_df['Action'].isin(event_types_of_interest)].copy()
event_df_filtered['Timestamp_dt'] = pd.to_datetime(
    event_df_filtered['Timestamp'], unit='ms'
)

# ─── 2. SETUP SNIPPET PARAMETERS ─────────────────────────────────────────────

window_size = 100
half_window = window_size // 2
save_root   = 'snippets'
os.makedirs(save_root, exist_ok=True)

# Pre‐extract the 6 “raw channels” into a (N×6) array and store timestamps:
raw_six    = imu_df[['accx','accy','accz','gyrox','gyroy','gyroz']].values
timestamps = imu_df['Timestamp_dt'].values

# ─── 3. HELPER: NEAREST INDEX ────────────────────────────────────────────────

def find_nearest_index(clicked_dt_float):
    """
    Convert Matplotlib float‐date (days since 0001‐01‐01) into pandas.Timestamp,
    then find the row index in imu_df whose Timestamp_dt is closest.
    """
    clicked_ts = pd.to_datetime(clicked_dt_float, unit='D', origin='unix')
    diffs = np.abs(timestamps.astype('datetime64[ns]') - np.datetime64(clicked_ts))
    return int(diffs.argmin())

# ─── 3.5. HIGHLIGHT CUT WINDOW ───────────────────────────────────────────────

highlight_lines = {'ax1': None, 'ax2': None}

def highlight_window(center_idx):
    """
    Draw vertical bars on ax1 and ax2 to show the window to be cut.
    Remove previous bars if present.
    """
    global highlight_lines
    for ax_key, ax in zip(['ax1', 'ax2'], [ax1, ax2]):
        # Remove old lines if they exist
        if highlight_lines[ax_key] is not None:
            for line in highlight_lines[ax_key]:
                line.remove()
        # Draw new lines at the beginning and end of the window (inclusive)
        left_idx = center_idx - half_window
        right_idx = center_idx + half_window - 1
        # Clamp indices to valid range
        left_idx = max(left_idx, 0)
        right_idx = min(right_idx, len(imu_df)-1)
        left_x = imu_df['Timestamp_dt'].iloc[left_idx]
        right_x = imu_df['Timestamp_dt'].iloc[right_idx]
        l1 = ax.axvline(left_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
        l2 = ax.axvline(right_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
        highlight_lines[ax_key] = (l1, l2)
    plt.draw()

def clear_highlight():
    global highlight_lines
    for ax_key, ax in zip(['ax1', 'ax2'], [ax1, ax2]):
        if highlight_lines[ax_key] is not None:
            for line in highlight_lines[ax_key]:
                line.remove()
            highlight_lines[ax_key] = None
    plt.draw()

def on_motion(event):
    # Only respond if mouse is over ax1 or ax2
    if event.inaxes not in (ax1, ax2):
        clear_highlight()
        return
    if event.xdata is None:
        clear_highlight()
        return
    center_idx = find_nearest_index(event.xdata)
    # Only highlight if window fits
    if center_idx < half_window or center_idx > len(imu_df) - half_window:
        clear_highlight()
        return
    highlight_window(center_idx)

# ─── 4. CALLBACK: ONCLICK ────────────────────────────────────────────────────

def onclick(event):
    # Only respond if click was in ax1 or ax2
    if event.inaxes not in (ax1, ax2):
        return

    if event.xdata is None:
        return

    center_idx = find_nearest_index(event.xdata)
    print(f"\nClicked on {'Mag A' if event.inaxes is ax1 else 'Mag G'} → index = {center_idx}")

    # Must have room on both sides for a 100-sample window
    if center_idx < half_window or center_idx > len(imu_df) - half_window:
        print(" ➔ Too close to start/end—cannot fit a full 100‐sample window.")
        clear_highlight()
        return

    # Highlight the window on click as well
    highlight_window(center_idx)

    # At this point, we need to show a small Tkinter window with five buttons.
    import tkinter as tk

    # Create a new top‐level window for label selection
    chooser = tk.Toplevel()
    chooser.title("Choose Label")
    # Make sure it pops up above the Matplotlib window:
    chooser.lift()
    chooser.attributes("-topmost", True)
    chooser.after_idle(chooser.attributes, "-topmost", False)

    # We'll define a helper to be called when a button is clicked.
    def make_save_callback(label):
        """
        Returns a function that, when called, will:
        1. Save the snippet under snippets/<label>/… 
        2. Destroy the chooser window.
        """
        def _save_and_close():
            # Create folder if needed
            folder = os.path.join(save_root, label)
            os.makedirs(folder, exist_ok=True)
            # Extract 100×6 snippet
            snippet = raw_six[center_idx - half_window : center_idx + half_window]
            # Count existing .npy in that label-folder
            existing = [
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and f.endswith('.npy')
            ]
            count = len(existing)
            fname = os.path.join(folder, f"{count:04d}.npy")
            np.save(fname, snippet)
            print(f" ➔ Saved snippet under label '{label}' → {fname}")
            chooser.destroy()
        return _save_and_close

    # List of five labels:
    labels = ["STANDING", "WALKING", "RUNNING", "PASS", "SHOOT"]

    # Place five buttons side by side in a single frame
    frame = tk.Frame(chooser)
    frame.pack(padx=10, pady=10)

    for lbl in labels:
        btn = tk.Button(
            frame,
            text=lbl,
            width=10,
            command=make_save_callback(lbl)
        )
        btn.pack(side="left", padx=5)

    # Make window size just large enough
    chooser.resizable(False, False)

    # Start the Tkinter event loop for this chooser window
    chooser.mainloop()

# ─── 5. PLOT EVERYTHING + CONNECT CLICK ────────────────────────────────────────

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 10))

# (a) Magnitude A
ax1.plot(imu_df['Timestamp_dt'], imu_df['mag_A'], label='Magnitude A')
ax1.set_ylabel('Mag A')
ax1.set_title('IMU Data + Events\n(click on Mag A or Mag G to snip)')
ax1.legend()
ax1.grid(True)

# (b) Magnitude G
ax2.plot(imu_df['Timestamp_dt'], imu_df['mag_G'], label='Magnitude G', color='orange')
ax2.set_ylabel('Mag G')
ax2.legend()
ax2.grid(True)

# (c) Events (shaded intervals + vertical dashed lines)
colors = plt.get_cmap('Dark2', len(event_types_of_interest)//2 + 1)
event_base_names = [
    e.split('_', 1)[1]
    for e in event_types_of_interest if e.startswith('INIZIO_')
]
event_base_color_map = {name: colors(i) for i, name in enumerate(event_base_names)}

# Shade INIZIO_… → FINE_… intervals
for name in event_base_names:
    skey = f"INIZIO_{name}"
    ekey = f"FINE_{name}"
    starts = event_df_filtered.loc[event_df_filtered['Action'] == skey, 'Timestamp_dt']
    ends   = event_df_filtered.loc[event_df_filtered['Action'] == ekey,   'Timestamp_dt']
    for s, e in zip(starts, ends):
        ax3.axvspan(s, e, color=event_base_color_map[name], alpha=0.15)
        midpoint = s + (e - s)/2
        ax3.text(
            midpoint, 0.5, name,
            color=event_base_color_map[name], fontsize=10,
            ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                      boxstyle='round,pad=0.2'),
            clip_on=True, transform=ax3.get_xaxis_transform()
        )

# Dashed vertical lines + labels
for _, row in event_df_filtered.iterrows():
    t      = row['Timestamp_dt']
    action = row['Action']
    base   = action.split('_', 1)[1] if '_' in action else action
    color  = event_base_color_map.get(base, 'black')
    ax3.axvline(x=t, color=color, linestyle='--', linewidth=1.2, alpha=0.8)
    ax3.text(
        t, 0.02, action, rotation=90, ha='left', va='bottom',
        color=color, fontsize=8, backgroundcolor='white',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7,
                  boxstyle='round,pad=0.1'),
        clip_on=True, transform=ax3.get_xaxis_transform()
    )

ax3.set_xlabel('Time')
ax3.set_ylim(0, 1)
ax3.set_yticks([])
ax3.grid(True)

plt.tight_layout()
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
plt.show()
