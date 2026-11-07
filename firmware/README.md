# Robot Firmware (ESP32)

Embedded firmware for the physical two-wheeled balancing robot (`ezspin_robot_balancing`).

## Overview

The firmware runs on an **ESP32** microcontroller, utilizing:
- **[SimpleFOC](https://docs.simplefoc.com/)**: Field-Oriented Control for dual BLDC gimbal motors (L6234 drivers).
- **MPU6050 with DMP (Digital Motion Processor)**: Hardware sensor fusion running DMP 6.12 at 100–200 Hz for drift-free pitch and gyro readings.
- **Embedded Neural Network Policy (`actor.h`)**: Pure C forward pass implementing the trained RL actor with zero dynamic memory allocation (`malloc`/`free`).
- **CAN Bus & Bluetooth Serial Interface**: Real-time telemetry, remote gamepad control (`cmd_vel`), and tuning commands.

---

## Directory Structure

```
firmware/
└── ezspin_robot_balancing/
    ├── ezspin_robot_balancing.ino   # Main Arduino sketch and control loop
    ├── actor.h                      # Exported pure-C neural network policy
    ├── balancer_pinouts.h           # ESP32 pin definitions for motor drivers & encoders
    ├── can.h / can.cpp              # CAN bus communication (telemetry & joystick cmd_vel)
    ├── imu.h / imu.cpp              # MPU6050 DMP driver and sensor caching
    └── src/
        ├── I2Cdev/                  # Fast I2C hardware bus interface
        └── MPU6050/                 # InvenSense MPU6050 DMP drivers
```

---

## Hardware & Pinout Setup

Default pin mapping for **ESP32 D1 R32** (see `balancer_pinouts.h`):

| Signal | ESP32 GPIO | Description |
|---|---|---|
| `MOT1_A`, `MOT1_B`, `MOT1_C` | GPIO 25, 12, 26 | Motor 1 (Left) PWM phases |
| `MOT1_EN` | GPIO 14 | Motor 1 enable pin |
| `MOT2_A`, `MOT2_B`, `MOT2_C` | GPIO 32, 13, 33 | Motor 2 (Right) PWM phases |
| `MOT2_EN` | GPIO 27 | Motor 2 enable pin |
| `I2C SDA` / `SCL` | GPIO 21 / GPIO 22 | MPU6050 6-DOF IMU |
| `CAN RX` / `TX` | GPIO 4 / GPIO 5 | SN65HVD230 / VP230 CAN transceiver |

---

## Deploying a New Trained Policy

When you train a new model in simulation, export it directly to `firmware/ezspin_robot_balancing/actor.h`:

```bash
# From the repository root:
python export_onnx.py runs/ppo_full/best_model.zip -o runs/ppo_full/best_model.onnx
python onnx_to_c.py runs/ppo_full/best_model.onnx firmware/ezspin_robot_balancing/actor.h
```

`actor.h` contains:
- Layer weights and biases formatted as static float arrays in Flash memory.
- `void actor_forward(const float obs[4], float action[2])`: deterministic inference executing in <100 microseconds on ESP32.

---

## Building and Flashing

1. Install [Arduino IDE](https://www.arduino.cc/en/software) with the **esp32** board package by Espressif.
2. Install library dependencies via Arduino Library Manager:
   - **SimpleFOC** (v2.3+)
3. Open `firmware/ezspin_robot_balancing/ezspin_robot_balancing.ino`.
4. Select Board: **ESP32 Dev Module** (or your specific ESP32 variant).
5. Compile and Upload via USB.

---

## Real-Time Tuning & Serial Commands

Connect via Serial Monitor (115200 baud) or Bluetooth Serial:

| Command | Action |
|---|---|
| `E` | Enable balancing control loop |
| `D` | Disable balancing control loop (motors freewheel) |
| `M` | Toggle motor polarity (`+1.0` / `-1.0`) |
| `G` | Toggle gyro sign (`+1.0` / `-1.0`) |
| `S<val>` | Adjust torque voltage scale (e.g. `S0.60`) |
| `B<val>` | Adjust velocity damping gain (e.g. `B0.005`) |
| `V<val>` | Adjust position hold stiffness (e.g. `V0.010`, `V0` to disable) |
| `IMU CFG ON` | Enable raw high-speed telemetry streaming |
