import pandas as pd
import numpy as np
import tensorflow as tf
import time

# 1. Load FFT-only dataset
df = pd.read_csv("labels_test_fft.csv")
X = df.drop(columns=["label"]).values.astype(np.float32)

# 2. Manual 80/20 split (same as training)
n = len(X)
indices = np.arange(n)
np.random.seed(42)
np.random.shuffle(indices)
split = int(n * 0.8)
test_idx = indices[split:]
X_test = X[test_idx]

# 3. Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mlp_fft_q.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 4. Prepare one test sample and quantize if needed
expected_shape = input_details[0]['shape']
# Print the expected shape for debugging (optional)
# print(f"Expected input shape: {expected_shape}")

# Reshape the sample to match the expected input shape
sample = X_test[0].reshape(expected_shape).astype(np.float32)
input_type = input_details[0]['dtype']
input_scale, input_zero_point = input_details[0]['quantization']

def quantize(x):
    return (x / input_scale + input_zero_point).astype(input_type)

if input_type == np.int8 or input_type == np.uint8:
    sample = quantize(sample)

# 5. Warm-up runs
for _ in range(10):
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()

# 6. Measure inference time
n_runs = 100
times = []
for _ in range(n_runs):
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    end = time.time()
    times.append(end - start)

mean_time = np.mean(times)
print(f"✅ Average inference time over {n_runs} runs: {mean_time*1000:.2f} ms")
