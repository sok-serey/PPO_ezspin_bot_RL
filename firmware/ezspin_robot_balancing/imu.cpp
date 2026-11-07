#include "imu.h"
#include "./src/I2Cdev/I2Cdev.h"
#include "./src/MPU6050/MPU6050_6Axis_MotionApps612.h"
#include "Wire.h"

// ── IMU instance ──────────────────────────────────────────────────────────────
MPU6050 mpu;

// ── Streaming flag ────────────────────────────────────────────────────────────
bool imuStreamActive = false;

// ── MPU control/status vars ───────────────────────────────────────────────────
bool     imuReady   = false;
uint8_t  mpuIntStatus;
uint8_t  devStatus;
uint16_t packetSize;
uint16_t fifoCount;
uint8_t  fifoBuffer[64];

// ── Orientation vars ──────────────────────────────────────────────────────────
Quaternion  q;
VectorFloat gravity;
float       ypr[3];   // [yaw, pitch, roll] radians

// ── Cache ─────────────────────────────────────────────────────────────────────
static float cached_roll       = 0.0f;
static float cached_pitch      = 0.0f;
static float cached_pitch_rate = 0.0f;
bool         cache_valid       = false;

// ── Serial stream timing ──────────────────────────────────────────────────────
static unsigned long lastPrintMs        = 0;
static const unsigned long STREAM_INTERVAL_MS = 20;  // 50 Hz

// ─────────────────────────────────────────────────────────────────────────────
//  Initialize IMU + DMP
//  Matches original imu_helpers.cpp init exactly — keeps CalibrateGyro,
//  sensor fusion gain adjustment, and 2 second startup delay
// ─────────────────────────────────────────────────────────────────────────────
int initIMU() {
    Wire.begin();
    Wire.setClock(400000);

    Serial.println(F("Initializing I2C devices..."));
    mpu.initialize();

    mpu.setClockSource(MPU6050_CLOCK_PLL_ZGYRO);
    mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_2000);
    mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_2);
    mpu.setSleepEnabled(false);

    Serial.println(F("Testing device connections..."));
    Serial.println(mpu.testConnection()
        ? F("MPU6050 connection successful")
        : F("MPU6050 connection failed"));

    Serial.println(F("Initializing DMP..."));
    devStatus = mpu.dmpInitialize();

    if (devStatus == 0) {
        // Only gyro calibration — matches original imu_helpers.cpp
        mpu.CalibrateGyro(6);
        Serial.println();
        mpu.PrintActiveOffsets();

        Serial.println(F("Enabling DMP..."));
        mpu.setDMPEnabled(true);

        Serial.println(F("DMP ready! Waiting for first interrupt..."));
        mpuIntStatus = mpu.getIntStatus();
        imuReady     = true;
        packetSize   = mpu.dmpGetFIFOPacketSize();

        Serial.print(F("DMP packet size: "));
        Serial.println(packetSize);
    } else {
        Serial.print(F("DMP Initialization failed (code "));
        Serial.print(devStatus);
        Serial.println(F(")"));
    }

    // Matches original — 2 second delay then sensor fusion gain adjustment
    delay(2000);
    Serial.println(F("Adjusting DMP sensor fusion gain..."));
    mpu.setMemoryBank(0);
    mpu.setMemoryStartAddress(0x60);
    mpu.writeMemoryByte(0);
    mpu.writeMemoryByte(0x20);
    mpu.writeMemoryByte(0);
    mpu.writeMemoryByte(0);

    return imuReady ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  hasDataIMU — check if new DMP packet is ready in FIFO
//  Matches original imu_helpers.cpp behaviour using dmpGetCurrentFIFOPacket
// ─────────────────────────────────────────────────────────────────────────────
int hasDataIMU() {
    if (!imuReady) return 0;
    return mpu.dmpGetCurrentFIFOPacket(fifoBuffer) ? 1 : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
//  updateIMUCache — read one DMP packet and update cached values
//  Call once per loop() — single point of FIFO access
//  Returns true if new data was read
// ─────────────────────────────────────────────────────────────────────────────
bool updateIMUCache() {
    if (!hasDataIMU()) return false;

    // Read quaternion from FIFO — same as original imu_helpers.cpp
    mpu.dmpGetQuaternion(&q, fifoBuffer);

    // ── Pitch — MUST match original imu_helpers.cpp getPitchIMU() exactly ────
    // Original uses this specific atan2 formula, NOT dmpGetYawPitchRoll ypr[1]
    // These give different angle conventions — using ypr[1] breaks balancing
    float pitch_new = -_PI_2 + atan2(
        q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
        2.0f * (q.y * q.z + q.w * q.x)
    );

    // Debouncing — identical to original
    if (abs(pitch_new - cached_pitch) > 0.1f)
        cached_pitch += _sign(pitch_new - cached_pitch) * 0.01f;
    else
        cached_pitch = pitch_new;

    // Read hardware gyroscope directly from DMP packet (no finite-difference lag)
    int16_t gyro_raw[3];
    mpu.dmpGetGyro(gyro_raw, fifoBuffer);
    // At MPU6050_GYRO_FS_2000: 16.4 LSB = 1 deg/s -> 939.65 LSB = 1 rad/s
    cached_pitch_rate = (float)gyro_raw[0] / 939.65f;

    // Roll and yaw — use dmpGetYawPitchRoll for completeness
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
    cached_roll = ypr[2];
    cache_valid = true;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Getters — return cached values (radians & rad/s)
//  Never touch FIFO directly
// ─────────────────────────────────────────────────────────────────────────────
float getRollIMU()      { return cached_roll;       }
float getPitchIMU()     { return cached_pitch;      }
float getYawIMU()       { return ypr[0];            }
float getPitchRateIMU() { return cached_pitch_rate; }

// ─────────────────────────────────────────────────────────────────────────────
//  loopIMU — call every loop() iteration
//  Updates cache + optional serial stream
// ─────────────────────────────────────────────────────────────────────────────
void loopIMU() {
    if (!imuReady) return;

    // Single FIFO read per loop — updates cached_roll, cached_pitch, ypr[]
    updateIMUCache();

    if (!imuStreamActive) return;

    unsigned long now = millis();
    if (now - lastPrintMs < STREAM_INTERVAL_MS) return;
    lastPrintMs = now;

    Serial.print(F("[IMU] roll: "));
    Serial.print(cached_roll, 4);
    Serial.print(F(" pitch: "));
    Serial.print(cached_pitch, 4);
    Serial.print(F(" yaw: "));
    Serial.println(ypr[0], 4);
}