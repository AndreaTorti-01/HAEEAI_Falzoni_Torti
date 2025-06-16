import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

# --- Step 1: Generate labels.csv from dataset folder ---
def generate_labels(snippets_dir, output_csv):
    samples = []
    labels = []
    for label in os.listdir(snippets_dir):
        label_folder = os.path.join(snippets_dir, label)
        if not os.path.isdir(label_folder):
            continue
        for fname in os.listdir(label_folder):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(label_folder, fname)
            data = np.load(fpath)
            sample = data.flatten()
            samples.append(sample)
            labels.append(label)
    X = np.vstack(samples)
    df = pd.DataFrame(X)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    print(f"✅ Dataset saved to {output_csv}")

def to_fft(csv_path, output_csv, window_len=100, num_channels=6):
    df = pd.read_csv(csv_path)
    labels = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    def get_fft_features(window_flat, win_len, n_channels):
        window = window_flat.reshape(win_len, n_channels)
        features = []
        for ch in range(n_channels):
            fft_vals = np.fft.fft(window[:, ch])
            mag = np.abs(fft_vals)
            phase = np.angle(fft_vals)
            features.append(mag)
            features.append(phase)
        return np.hstack(features)
    X_fft = np.array([get_fft_features(row, window_len, num_channels) for row in X])
    df_fft = pd.DataFrame(X_fft)
    df_fft["label"] = labels
    df_fft.to_csv(output_csv, index=False)
    print(f"✅ Saved FFT-only features to {output_csv}")

def train_model():
    df = pd.read_csv("labels_fft.csv")
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values
    num_timesteps = 100
    if X.shape[1] % num_timesteps != 0:
        raise ValueError(f"Number of columns ({X.shape[1]}) is not divisible by {num_timesteps}.")
    num_features = X.shape[1] // num_timesteps
    X = X.reshape(-1, num_timesteps, num_features)
    classes = sorted(set(y.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx[label] for label in y])
    y_onehot = tf.keras.utils.to_categorical(y_indices, num_classes=len(classes))
    def build_model(num_classes):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(num_timesteps, num_features)),
            tf.keras.layers.SeparableConv1D(8, 3, padding='same', activation='relu'),
            tf.keras.layers.SeparableConv1D(8, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    val_accuracies = []
    print("Starting cross-validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_indices), 1):
        print(f"Training fold {fold}/{k}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]
        model = build_model(len(classes))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )
        history = model.fit(
            X_train, y_train, 
            epochs=100, 
            batch_size=4, 
            verbose=0, 
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        val_acc = max(history.history['val_accuracy'])
        val_accuracies.append(val_acc)
        print(f"Fold {fold} validation accuracy: {val_acc:.4f}")
    mean_acc = np.mean(val_accuracies)
    std_acc = np.std(val_accuracies)
    print(f"✅ Cross-validation mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print("Training final model on all data...")
    final_model = build_model(len(classes))
    final_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping_final = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy', patience=15, restore_best_weights=True
    )
    history = final_model.fit(
        X, y_onehot, 
        epochs=100, 
        batch_size=4, 
        verbose=1,
        callbacks=[early_stopping_final]
    )
    final_acc = max(history.history['accuracy'])
    print(f"✅ Final training accuracy: {final_acc:.4f}")
    final_model.save("mlp_imu_classifier_fft.h5")
    print("✅ Model saved as mlp_imu_classifier_fft.h5")

def quantize_model():
    df = pd.read_csv("labels_fft.csv")
    X_train = df.drop(columns=["label"]).values.astype(np.float32)
    model = tf.keras.models.load_model("mlp_imu_classifier_fft.h5")
    def representative_data_gen():
        input_shape = model.input_shape
        for i in range(min(100, len(X_train))):
            sample = X_train[i]
            reshaped = sample.reshape((1,) + input_shape[1:])
            yield [reshaped.astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    with open("mlp_fft_q.tflite", "wb") as f:
        f.write(tflite_quant_model)
    print("✅ Quantized model saved as mlp_fft_q.tflite")

def test_quantized_model():
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    df = pd.read_csv("labels_test_fft.csv")
    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values
    classes = sorted(set(y.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_indices = np.array([class_to_idx[label] for label in y])
    X_test = X
    y_test = y_indices
    interpreter = tf.lite.Interpreter(model_path="mlp_fft_q.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_type = input_details[0]['dtype']
    input_scale, input_zero_point = input_details[0]['quantization']
    def quantize(x):
        return (x / input_scale + input_zero_point).astype(input_type)
    y_pred = []
    for x in X_test:
        x_in = x.reshape(input_details[0]['shape'])
        if input_type == np.int8 or input_type == np.uint8:
            x_in = quantize(x_in)
        interpreter.set_tensor(input_details[0]['index'], x_in)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output))
    accuracy = (np.array(y_pred) == y_test).mean()
    print(f"✅ Quantized TFLite model test accuracy: {accuracy:.2%}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
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

if __name__ == "__main__":
    # Step 1: Generate labels.csv and labels_test.csv
    generate_labels("dataset", "labels.csv")
    generate_labels("dataset_test", "labels_test.csv")
    # Step 2: Convert to FFT
    to_fft("labels.csv", "labels_fft.csv")
    to_fft("labels_test.csv", "labels_test_fft.csv")
    # Step 3: Train model
    train_model()
    # Step 4: Quantize model
    quantize_model()
    # Step 5: Test quantized model
    test_quantized_model()
    print("✅ All steps completed.")
