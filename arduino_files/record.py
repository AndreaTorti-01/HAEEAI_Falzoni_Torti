import serial
from datetime import datetime
import time

ser = serial.Serial('/dev/ttyACM0', 115200)
start_time = datetime.now()
start_str = start_time.strftime("%Y%m%d_%H%M%S")
filename = f"imu_data_{start_str}.txt"
with open(filename, 'w') as f:
    last_ping = time.time()
    # Skip the first (dirty) line
    ser.readline()
    while True:
        line = ser.readline().decode('utf-8').rstrip()
        now = datetime.now()
        unix_time_ms = int(now.timestamp() * 1000)
        # Write unix time in ms, then the data, separated by tab
        f.write(f"{unix_time_ms}\t{line}\n")
        current_time = time.time()
        if current_time - last_ping >= 0.2:
            print("Recording...")  # status message
            last_ping = current_time