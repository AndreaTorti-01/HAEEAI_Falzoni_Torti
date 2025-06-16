import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools

# Set the dataset folder here (should contain 'test' subfolder)
DATA_DIR = 'dataset_left'  # <-- change as needed
ACCEL_SCALE = 16.0
GYRO_SCALE = 2000.0
KERAS_MODEL_NAME = 'norm_football_model.keras'
TFLITE_MODEL_NAME = 'norm_football_model.tflite'

SEED = 99
np.random.seed(SEED)


def load_split(split):
    X, y = [], []
    split_dir = os.path.join(DATA_DIR, split)
    for label in sorted(os.listdir(split_dir)):
        lbl_dir = os.path.join(split_dir, label)
        if not os.path.isdir(lbl_dir): continue
        for fn in os.listdir(lbl_dir):
            if not fn.endswith('.npy'): continue
            arr = np.load(os.path.join(lbl_dir, fn)).astype('float32')
            arr[:, :3] /= ACCEL_SCALE
            arr[:, 3:] /= GYRO_SCALE
            X.append(arr)
            y.append(label)
    return np.stack(X), np.array(y)

def plot_confusion(cm, classes, title):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
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

def keras_predict(model, X):
    y_pred = model.predict(X)
    return np.argmax(y_pred, axis=1)

def tflite_predict(interpreter, X):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    y_pred = []
    for i in range(X.shape[0]):
        x = X[i:i+1]
        x_q = np.round(x / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], x_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(output_details[0]['index'])
        out_f = (out_q.astype(np.float32) - output_zero_point) * output_scale
        y_pred.append(np.argmax(out_f))
    return np.array(y_pred)

def main():
    # Load test data
    X_test, y_test = load_split('test')
    # Fit label encoder on test labels only
    le = LabelEncoder().fit(y_test)
    y_test_i = le.transform(y_test)
    num_classes = len(le.classes_)

    # --- Keras model ---
    keras_model = tf.keras.models.load_model(KERAS_MODEL_NAME)
    y_pred_keras = keras_predict(keras_model, X_test)
    acc_keras = (y_pred_keras == y_test_i).mean()
    cm_keras = confusion_matrix(y_test_i, y_pred_keras)
    print(f"Keras model accuracy: {acc_keras*100:.2f}%")
    plot_confusion(cm_keras, le.classes_, title="Keras Model Confusion Matrix")

    # --- TFLite model ---
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_NAME)
    interpreter.allocate_tensors()
    y_pred_tflite = tflite_predict(interpreter, X_test)
    acc_tflite = (y_pred_tflite == y_test_i).mean()
    cm_tflite = confusion_matrix(y_test_i, y_pred_tflite)
    print(f"TFLite quantized model accuracy: {acc_tflite*100:.2f}%")
    plot_confusion(cm_tflite, le.classes_, title="TFLite Quantized Model Confusion Matrix")

if __name__ == '__main__':
    main()
