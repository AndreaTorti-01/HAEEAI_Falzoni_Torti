/*
cd lib
git clone https://github.com/tensorflow/tflite-micro-arduino-examples Arduino_TensorFlowLite
*/

/*
be sure to modify .pio\libdeps\nano33ble\Arduino_LSM9DS1\src\LSM9DS1.cpp
line   writeRegister(LSM9DS1_ADDRESS, LSM9DS1_CTRL_REG6_XL, 0x70); // 119 Hz, 4g
should become   writeRegister(LSM9DS1_ADDRESS, LSM9DS1_CTRL_REG6_XL, 0x68); // 119 Hz, 16g

also, lines
  x = data[0] * 4.0 / 32768.0;
  y = data[1] * 4.0 / 32768.0;
  z = data[2] * 4.0 / 32768.0;

in readAcceleration() should become
  x = data[0] * 4.0 / 32768.0 / 0.163;
  y = data[1] * 4.0 / 32768.0 / 0.163;
  z = data[2] * 4.0 / 32768.0 / 0.163;
*/

#undef abs // allow std::abs in TensorFlow Lite headers
#include "norm_football_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino_LSM9DS1.h>

namespace
{
// model & interpreter
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
// arena
constexpr int kArenaSize = 20 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];
// data buffer: 100 samples × 6 channels
constexpr int kSamples = 100, kCh = 6;
constexpr int kWindow = 50; // half-second window
static float data_buf[kSamples * kCh];
// compact labels (order must match training data alphabetical sort)
const char *labels[] = {"P", "R", "H", "S", "W"}; // PASS, RUNNING, SHOOT, STANDING, WALKING
// quantization parameters
float input_scale = 0.0f;
int input_zero_point = 0;
float output_scale = 0.0f;
int output_zero_point = 0;
} // namespace

void setup()
{
    Serial.begin(115200);
    while (!Serial)
    {
    }
    if (!IMU.begin())
    {
        Serial.println("IMU init failed");
        while (1)
        {
        }
    }
    tflite::InitializeTarget();
    model = tflite::GetModel(norm_football_model_tflite);
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kArenaSize);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        Serial.println("Tensor alloc failed");
        while (1)
        {
        }
    }
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Get quantization parameters
    input_scale = input->params.scale;
    input_zero_point = input->params.zero_point;
    output_scale = output->params.scale;
    output_zero_point = output->params.zero_point;

    Serial.println("Model loaded successfully");
}

void loop()
{
    // Slide previous 50 samples to the front
    memmove(data_buf, data_buf + kWindow * kCh, (kSamples - kWindow) * kCh * sizeof(float));

    // Collect 50 new samples (0.5 s @ 100 Hz)
    for (int i = kSamples - kWindow; i < kSamples; ++i)
    {
        uint32_t t0 = millis();
        float ax, ay, az, gx, gy, gz;
        while (!IMU.accelerationAvailable())
        {
        }
        IMU.readAcceleration(ax, ay, az);
        while (!IMU.gyroscopeAvailable())
        {
        }
        IMU.readGyroscope(gx, gy, gz);
        // normalize: ±16 g and ±2000 dps
        data_buf[i * kCh + 0] = ax / 16.0f;
        data_buf[i * kCh + 1] = ay / 16.0f;
        data_buf[i * kCh + 2] = az / 16.0f;
        data_buf[i * kCh + 3] = gx / 2000.0f;
        data_buf[i * kCh + 4] = gy / 2000.0f;
        data_buf[i * kCh + 5] = gz / 2000.0f;
        // maintain 100 Hz
        uint32_t dt = millis() - t0;
        if (dt < 10) {
            delay(10 - dt);
        }
    }

    // copy into input tensor with quantization (int8 model expects quantized input)
    for (int i = 0; i < kSamples * kCh; ++i)
    {
        // Quantize: value = round(float_value / scale + zero_point)
        int quantized = round(data_buf[i] / input_scale + input_zero_point);
        // Clamp to int8 range
        quantized = max(-128, min(127, quantized));
        input->data.int8[i] = (int8_t)quantized;
    }

    // inference
    if (interpreter->Invoke() != kTfLiteOk)
    {
        Serial.println("Invoke error");
        return;
    }

    // find best class (dequantize output first)
    int classes = output->dims->data[1];
    float best = (output->data.int8[0] - output_zero_point) * output_scale;
    int pred = 0;
    for (int i = 1; i < classes; ++i)
    {
        float val = (output->data.int8[i] - output_zero_point) * output_scale;
        if (val > best)
        {
            best = val;
            pred = i;
        }
    }

    // report
    Serial.println(labels[pred]);
}
