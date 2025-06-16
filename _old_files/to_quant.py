import pandas as pd
import numpy as np
import tensorflow as tf

# 1. Load FFT-only training dataset from 'dataset' folder
df = pd.read_csv("labels_fft.csv")
X_train = df.drop(columns=["label"]).values.astype(np.float32)

# 2. Load your trained model
model = tf.keras.models.load_model("mlp_imu_classifier_fft.h5")

# 3. Representative dataset generator
def representative_data_gen():
    input_shape = model.input_shape  # e.g., (None, height, width, channels)
    for i in range(min(100, len(X_train))):
        sample = X_train[i]
        # Reshape sample to (1, height, width, channels)
        reshaped = sample.reshape((1,) + input_shape[1:])
        yield [reshaped.astype(np.float32)]

# 4. TFLite converter with full integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# 5. Convert and save
tflite_quant_model = converter.convert()
with open("mlp_fft_q.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("✅ Quantized model saved as mlp_fft_q.tflite")
