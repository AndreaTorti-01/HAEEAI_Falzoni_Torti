// Include the necessary library for the LSM9DS1 sensor.
// This library is typically pre-installed with the Arduino Nano 33 BLE Sense board package.
#include <Arduino_LSM9DS1.h> // Correct library for Arduino Nano 33 BLE Sense IMU

void setup()
{
    Serial.begin(115200); // Initialize serial communication at 115200 baud rate

    // Wait for serial port to connect. Needed for native USB port only.
    while (!Serial)
    {
        delay(10);
    }

    // Initialize the LSM9DS1 sensor.
    if (!IMU.begin())
    {
        Serial.println("Failed to initialize IMU!");
        // Send a message that Teleplot won't try to plot, but will show in console
        Serial.println("!ERROR: IMU not found.");
        while (1)
        {
            delay(100); // Halt program execution if sensor not found
        }
    }

    // Set accelerometer range to ±16g and gyroscope range to ±2000°/s
    // Note: Arduino_LSM9DS1 library may not expose range setting functions
    // These ranges are typically the default or maximum for LSM9DS1
}

void loop()
{
    uint32_t start = millis();

    float xAcc, yAcc, zAcc;
    float xGyro, yGyro, zGyro;

    // Read accelerometer data
    if (IMU.accelerationAvailable())
    {
        IMU.readAcceleration(xAcc, yAcc, zAcc);
    }

    // Read gyroscope data
    if (IMU.gyroscopeAvailable())
    {
        IMU.readGyroscope(xGyro, yGyro, zGyro);
    }

    // Print in Teleplot format
    Serial.print(">xAcc:");
    Serial.println(xAcc);
    Serial.print(">yAcc:");
    Serial.println(yAcc);
    Serial.print(">zAcc:");
    Serial.println(zAcc);
    Serial.print(">xGyro:");
    Serial.println(xGyro);
    Serial.print(">yGyro:");
    Serial.println(yGyro);
    Serial.print(">zGyro:");
    Serial.println(zGyro);

    // Ensure 100Hz sample rate (10ms per sample)
    uint32_t elapsed = millis() - start;
    if (elapsed < 10)
    {
        delay(10 - elapsed);
    }
}