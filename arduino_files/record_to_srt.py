INPUT_FILE = "football_data/inference_results_20250616_151624.txt"

import os
from datetime import timedelta

def ms_to_srt_time(ms):
    td = timedelta(milliseconds=ms)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{td.days*24+hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def convert_txt_to_srt(input_file):
    # Read lines and parse
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    entries = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            timestamp, label = parts
            entries.append((int(timestamp), label))
    if not entries:
        print("No valid entries found.")
        return
    base_time = entries[0][0]
    label_map = {
        "S": "Standing",
        "P": "Pass!",
        "R": "Running",
        "W": "Walking",
        "H": "Shot!"
    }
    # Generate SRT
    srt_lines = []
    for i, (start, label) in enumerate(entries):
        # Subtract base_time to start from zero
        start_rel = start - base_time
        end = entries[i+1][0] if i+1 < len(entries) else start + 1000
        end_rel = end - base_time
        label_full = label_map.get(label, label)
        srt_lines.append(f"{i+1}")
        srt_lines.append(f"{ms_to_srt_time(start_rel)} --> {ms_to_srt_time(end_rel)}")
        srt_lines.append(label_full)
        srt_lines.append("")
    # Write SRT file
    out_file = os.path.splitext(input_file)[0] + ".srt"
    with open(out_file, "w") as f:
        f.write("\n".join(srt_lines))
    print(f"SRT file written to {out_file}")

# Example usage:
if __name__ == "__main__":
    convert_txt_to_srt(INPUT_FILE)