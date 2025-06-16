# transforms imu data into the new format

input_file = "football_data/imu_data.txt"
output_file = "football_data/imu_data_20250529_142014.txt"

# The first unix timestamp in milliseconds
first_unix_ms = 1748521214624

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    prev_sec = None
    prev_ms = None
    unix_ms = first_unix_ms

    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            sec_str, data = line.split(",", 1)
            sec = float(sec_str)
        except ValueError:
            continue  # skip malformed lines

        ms = int(sec * 1000)
        if prev_ms is None:
            ms_delta = ms
        else:
            ms_delta = ms - prev_ms

        # Advance unix_ms by ms_delta
        unix_ms += ms_delta
        unix_time = unix_ms // 1000  # integer unix time in seconds

        fout.write(f"{ms},{unix_time},{data}\n")

        prev_sec = sec
        prev_ms = ms