import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

# 1. Load the FFT-only dataset
df = pd.read_csv("labels_fft.csv")

# 2. Prepare features and labels
X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values

# Dynamically determine number of features per timestep
num_timesteps = 100
if X.shape[1] % num_timesteps != 0:
    raise ValueError(f"Number of columns ({X.shape[1]}) is not divisible by {num_timesteps}.")
num_features = X.shape[1] // num_timesteps
X = X.reshape(-1, num_timesteps, num_features)

# 3. Encode labels
classes = sorted(set(y.tolist()))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_indices = np.array([class_to_idx[label] for label in y])
y_onehot = tf.keras.utils.to_categorical(y_indices, num_classes=len(classes))

# MODEL SUGGERITO: Depthwise separable Conv1D + GlobalAveragePooling
def build_model(num_classes):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_timesteps, num_features)),
        tf.keras.layers.SeparableConv1D(8, 3, padding='same', activation='relu'),
        tf.keras.layers.SeparableConv1D(8, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

# 4. Cross-validation (for evaluation only)
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True)  # removed random_state=42
val_accuracies = []

print("Starting cross-validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_indices), 1):
    print(f"Training fold {fold}/{k}...")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

    model = build_model(len(classes))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Add early stopping to prevent overfitting
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

# Train final model on all data
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
