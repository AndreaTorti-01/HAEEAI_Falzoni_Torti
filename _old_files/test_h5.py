import pandas as pd
import numpy as np
import tensorflow as tf

# 1. Load FFT-only dataset
df = pd.read_csv("labels_test.csv")
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

# Reshape X_test to (num_samples, 100, 6) as required by the model
try:
    X_test = X_test.reshape(-1, 100, 6)
except Exception as e:
    raise ValueError(f"Could not reshape X_test to (-1, 100, 6): {e}")

# 4. Load Keras .h5 model
model = tf.keras.models.load_model("mlp_imu_classifier_fft.h5")

# 5. Inference
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# 6. Accuracy
accuracy = (y_pred == y_test).mean()
print(f"✅ Keras .h5 model test accuracy: {accuracy:.2%}")

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
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
