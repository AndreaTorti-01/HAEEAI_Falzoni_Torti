import pandas as pd
import numpy as np
import tensorflow as tf
# import sklearn if needed (correct import is below in the try block)

# 1. Load FFT-only dataset
df = pd.read_csv("labels_test_fft.csv")
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

# 2. Label encoding (for accuracy evaluation)
classes = sorted(set(y.tolist()))
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}
y_indices = np.array([class_to_idx[label] for label in y])

# 3. Use all data for testing
X_test = X
y_test = y_indices

# 4. Load quantized TFLite model
interpreter = tf.lite.Interpreter(model_path="mlp_fft_q.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 5. Helper: quantize input if needed
input_type = input_details[0]['dtype']
input_scale, input_zero_point = input_details[0]['quantization']

def quantize(x):
    return (x / input_scale + input_zero_point).astype(input_type)

# 6. Inference loop
y_pred = []
for x in X_test:
    # Reshape input to match model's expected input shape
    x_in = x.reshape(input_details[0]['shape'])
    if input_type == np.int8 or input_type == np.uint8:
        x_in = quantize(x_in)
    interpreter.set_tensor(input_details[0]['index'], x_in)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(np.argmax(output))

# 7. Accuracy
accuracy = (np.array(y_pred) == y_test).mean()
print(f"✅ Quantized TFLite model test accuracy: {accuracy:.2%}")

# Add confusion matrix
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    print("scikit-learn is required for confusion matrix.")
else:
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # Show confusion matrix with matplotlib
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    # Use class labels from CSV
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    # Annotate cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
