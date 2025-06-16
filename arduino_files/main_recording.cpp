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

    // Print in format: xacc	yacc	zacc	xgyro	ygyro	zgyro
    Serial.println(String(xAcc, 6) + '\t' + String(yAcc, 6) + '\t' + String(zAcc, 6) + '\t' + String(xGyro, 6) + '\t' +
                   String(yGyro, 6) + '\t' + String(zGyro, 6));

    // Ensure 100Hz sample rate (10ms per sample)
    uint32_t elapsed = millis() - start;
    if (elapsed < 10)
    {
        delay(10 - elapsed);
    }
}