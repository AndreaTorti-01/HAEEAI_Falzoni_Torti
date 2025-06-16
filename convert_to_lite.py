import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools

def representative_data_gen():
    # Load a small sample of training data for calibration
    DATA_DIR = 'dataset'
    ACCEL_SCALE = 16.0
    GYRO_SCALE = 2000.0
    X = []
    split_dir = os.path.join(DATA_DIR, 'train')
    # Set seed for reproducibility
    SEED = 99
    np.random.seed(SEED)
    all_files = []
    for label in sorted(os.listdir(split_dir)):
        lbl_dir = os.path.join(split_dir, label)
        if not os.path.isdir(lbl_dir): continue
        for fn in os.listdir(lbl_dir):
            if fn.endswith('.npy'):
                all_files.append((os.path.join(lbl_dir, fn), label))
    # Shuffle and select up to 100 samples
    np.random.shuffle(all_files)
    for file_path, _ in all_files[:100]:
        arr = np.load(file_path).astype('float32')
        arr[:, :3] /= ACCEL_SCALE
        arr[:, 3:] /= GYRO_SCALE
        yield [np.expand_dims(arr, axis=0)]

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

def save_tflite_as_c_array(tflite_path, h_path, var_name="norm_football_model_tflite"):
    with open(tflite_path, "rb") as f:
        data = f.read()
    array_len = len(data)
    with open(h_path, "w") as f:
        f.write(f"const unsigned int {var_name}_len = {array_len};\n")
        f.write(f"const unsigned char {var_name}[] = {{\n")
        for i, b in enumerate(data):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x},")
            if (i + 1) % 12 == 0:
                f.write("\n")
            else:
                f.write(" ")
        if array_len % 12 != 0:
            f.write("\n")
        f.write(f"}};\n")

def main():
    # 1) Load the Keras model
    model = tf.keras.models.load_model('norm_football_model.keras')

    # 2) Convert to TFLite with 8-bit quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen  # <-- fix: no ()
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    # 3) Save the flatbuffer
    tflite_path = 'norm_football_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Saved quantized TFLite model to '{tflite_path}'")

    # 4) Emit C array as header (cross-platform, no xxd)
    h_path = 'norm_football_model.h'
    try:
        save_tflite_as_c_array(tflite_path, h_path)
        print(f"Generated C header at '{h_path}'")
    except Exception as e:
        print("Error: Failed to generate C header.")
        print(e)
        h_path = None

    # Print C array size in KB (actual array size, not header file size)
    array_size = None
    if h_path is not None and os.path.exists(h_path):
        with open(h_path, 'r') as f:
            for line in f:
                if line.strip().startswith('unsigned int') and '_len' in line:
                    array_size = int(line.strip().split('=')[1].replace(';','').strip())
                    break
    if array_size is not None:
        size_kb = array_size / 1024
        print(f"C array data size: {size_kb:.2f} KB")
    else:
        print("Could not determine C array size from header.")

    # 5) Evaluate quantized model on test set and print confusion matrix
    # Load test data and label encoder (copy logic from norm_train.py)
    DATA_DIR = 'dataset'
    ACCEL_SCALE = 16.0
    GYRO_SCALE = 2000.0
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
    X_test, y_test = load_split('test')

    # Load label encoder from training set
    X_train_full, y_train_full = load_split('train')
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_train_full)
    y_test_i = le.transform(y_test)
    num_classes = len(le.classes_)

    # Run inference with TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Quantization params
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    y_pred = []
    for i in range(X_test.shape[0]):
        x = X_test[i:i+1]
        # Quantize input
        x_q = np.round(x / input_scale + input_zero_point).astype(np.int8)
        interpreter.set_tensor(input_details[0]['index'], x_q)
        interpreter.invoke()
        out_q = interpreter.get_tensor(output_details[0]['index'])
        # Dequantize output
        out_f = (out_q.astype(np.float32) - output_zero_point) * output_scale
        y_pred.append(np.argmax(out_f))
    y_pred = np.array(y_pred)

    # Print accuracy percentage
    accuracy = (y_pred == y_test_i).mean()
    print(f"Quantized model accuracy: {accuracy*100:.2f}%")

    cm = confusion_matrix(y_test_i, y_pred)
    print("Confusion matrix for quantized model:")
    plot_confusion(cm, le.classes_)

if __name__ == '__main__':
    main()
