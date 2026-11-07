#ifndef IMU_H
#define IMU_H

#include "Arduino.h"

#define _PI_2 1.57079632679
#define _sign(a) ( ( (a) < 0 ) ? -1 : ( (a) > 0 ) )

// ── Init & loop ───────────────────────────────────
int  initIMU();
void loopIMU();

// ── Data availability ─────────────────────────────
int  hasDataIMU();

// ── Single-call cache update (call once per loop) ─
bool updateIMUCache();

// ── Getters (return cached values, radians & rad/s) ───────
float getRollIMU();
float getPitchIMU();
float getYawIMU();
float getPitchRateIMU(); // Instantaneous physical gyro pitch rate (rad/s)

// ── Cache state ───────────────────────────────────
extern bool cache_valid;

// ── Serial streaming ──────────────────────────────
// Toggled by "IMU CFG ON" / "IMU CFG OFF"
// Prints: [IMU] roll: X.XXXX pitch: X.XXXX yaw: X.XXXX at ~50Hz
extern bool imuStreamActive;

#endif