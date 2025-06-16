import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools
import random

# Set seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = 'dataset'
ACCEL_SCALE = 16.0    # g range ±16
GYRO_SCALE  = 2000.0  # dps range ±2000

def load_split(split):
    X, y = [], []
    split_dir = os.path.join(DATA_DIR, split)
    for label in sorted(os.listdir(split_dir)):
        lbl_dir = os.path.join(split_dir, label)
        if not os.path.isdir(lbl_dir): continue
        for fn in os.listdir(lbl_dir):
            if not fn.endswith('.npy'): continue
            arr = np.load(os.path.join(lbl_dir, fn)).astype('float32')
            # normalize channels: first 3 by ACCEL_SCALE, next 3 by GYRO_SCALE
            arr[:, :3] /= ACCEL_SCALE
            arr[:, 3:] /= GYRO_SCALE
            X.append(arr)
            y.append(label)
    return np.stack(X), np.array(y)

def plot_confusion(cm, classes):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

def main():
    # load data
    X_train, y_train = load_split('train')
    X_val, y_val     = load_split('validation')
    X_test, y_test   = load_split('test')
    # encode labels
    le = LabelEncoder().fit(np.concatenate([y_train, y_val]))
    y_train_i = le.transform(y_train)
    y_val_i   = le.transform(y_val)
    y_test_i  = le.transform(y_test)
    num_classes = len(le.classes_)

    # build model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(16, 3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # train
    model.fit(X_train, y_train_i,
              validation_data=(X_val, y_val_i),
              epochs=100, batch_size=32,
              callbacks=[early_stop])
    # evaluate
    loss, acc = model.evaluate(X_test, y_test_i, verbose=0)
    print(f"Test accuracy: {acc*100:.2f}%")

    # confusion matrix
    y_pred = model.predict(X_test).argmax(axis=1)
    cm = confusion_matrix(y_test_i, y_pred)
    plot_confusion(cm, le.classes_)

    # save Keras model
    model.save('norm_football_model.keras')
    print("Model saved to 'norm_football_model.keras'")


if __name__ == '__main__':
    main()