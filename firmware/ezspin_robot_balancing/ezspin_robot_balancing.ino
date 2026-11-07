#include <SimpleFOC.h>
#include "balancer_pinouts.h"
#include "imu.h"
#include "can.h"
#include "BluetoothSerial.h"
#include <Preferences.h>
#include "actor.h"

BluetoothSerial bluetooth;

// ─────────────────────────────────────────────────────────────────────────────
// BALANCING ENABLE FLAG
// Set via CAN ID 0x010 (defined in can.h):
//   data[0] = 0x01 → enable  (R2 on gamepad)
//   data[0] = 0x00 → disable (L2 on gamepad)
// Serial command: 'E' = enable, 'D' = disable
// ─────────────────────────────────────────────────────────────────────────────
volatile bool balancing_enabled = true;

// ─────────────────────────────────────────────────────────────────────────────
// JOYSTICK cmd_vel — received via CAN ID 0x020
//   cmd_vel_linear  = linear.x  [m/s]  — forward/back
//   cmd_vel_angular = angular.z [rad/s] — turn left/right
// ─────────────────────────────────────────────────────────────────────────────
volatile float cmd_vel_linear  = 0.0f;
volatile float cmd_vel_angular = 0.0f;

// Scale factor for turning
#define CMD_VEL_ANGULAR_SCALE 1.5f   // Turn voltage difference between motors

// ─────────────────────────────────────────────────────────────────────────────
// REINFORCEMENT LEARNING SIM-TO-REAL CONFIGURATION
// ─────────────────────────────────────────────────────────────────────────────
// 1. Sim Equilibrium Offset:
//    The policy was trained in MuJoCo where the CAD robot balanced at:
//    sim_pitch_eq = 0.310 rad (~17.8 degrees) due to battery weight behind axle.
//    On your real robot, standing vertically is pitch = 0.0 rad.
//    Adding sim_pitch_eq maps vertical (0.0) -> policy equilibrium (0.310 rad, 0 torque).
float sim_pitch_eq = 0.310f; // radians

// 2. Motor Polarity:
//    Confirmed by hardware test: motor_sign = 1.0f drives wheels forward to balance!
//    Toggle with 'M' in Serial/Bluetooth.
float motor_sign   = 1.0f;

// 3. Gyroscope sign:
//    Physical gyro from MPU6050 eliminates finite-difference derivative lag.
//    MPU6050 X-gyro outputs negative rate when pitching forward.
//    Setting gyro_sign = -1.0f converts it to POSITIVE pitch rate in MuJoCo convention,
//    providing correct damping that prevents oscillation! Toggle with 'G'.
float gyro_sign    = -1.0f;

// 4. Voltage / Gain scaling & Filtering:
//    torque_scale = 0.60f tuned & confirmed on hardware for rock-solid, smooth balance without oscillation!
//    Adjustable in real-time via Serial/Bluetooth with 'S<val>' if needed.
const float MAX_VOLTAGE    = 6.0f;   // Max voltage clamp (Volts)
float       torque_scale   = 0.60f;  // Hardware-verified 60% power — zero oscillation!
float       action_lpf     = 0.65f;  // Action filter (0.65 = 65% new, 35% prev) kills high-frequency buzzing/chatter
const float MAX_FALL_ANGLE = 0.60f;  // ~34.4 degrees: emergency cutoff if robot falls over
float       pitch_trim     = 0.0f;   // Real robot physical trim offset in radians

// 5. Active Position & Velocity Hold Loop:
//    Integral on velocity = displacement dx (acts as a firm virtual spring back to spot)
//    Proportional on velocity = damping brake (prevents overshoot/hunting)
float vel_integral  = 0.0f;   // Integral of velocity (distance)
float vel_gain_p    = 0.005f; // Velocity damping brake (adjust with 'B<val>')
float vel_gain_i    = 0.010f; // Position hold stiffness (safe default 0.010; 'V0' disables, adjust with 'V<val>')
float last_avg_vel  = 0.0f;   // Telemetry: average wheel velocity (rad/s)
float last_vel_trim = 0.0f;   // Telemetry: velocity brake trim angle (rad)

// Telemetry state
float last_action_left  = 0.0f;
float last_action_right = 0.0f;
float last_pitch_rate   = 0.0f;

// Wheel geometry & origin tracking for 5-observation stay-in-place policy
#define WHEEL_RADIUS 0.034f // 68 mm wheel diameter
float start_shaft_angle1 = 0.0f;
float start_shaft_angle2 = 0.0f;
float current_pos_x      = 0.0f;

// ─────────────────────────────────────────────────────────────────────────────
// ENCODERS & MOTORS
// ─────────────────────────────────────────────────────────────────────────────
MagneticSensorSPI encoder1 = MagneticSensorSPI(AS5147_SPI, 15);
MagneticSensorSPI encoder2 = MagneticSensorSPI(AS5147_SPI, 16);

BLDCMotor motor  = BLDCMotor(11);
BLDCMotor motor2 = BLDCMotor(11);
BLDCDriver3PWM driver1 = BLDCDriver3PWM(MOT1_A, MOT1_B, MOT1_C, MOT1_EN);
BLDCDriver3PWM driver2 = BLDCDriver3PWM(MOT2_A, MOT2_B, MOT2_C, MOT2_EN);

// ─────────────────────────────────────────────────────────────────────────────
// CAN motor targets — defined here, extern'd in can.h
// ─────────────────────────────────────────────────────────────────────────────
float       target_position  = 0.0f;
float       target_velocity  = 0.0f;
float       target_torque    = 0.0f;
MessageType current_mode     = CONTROL_TORQUE;

float       target_position2 = 0.0f;
float       target_velocity2 = 0.0f;
float       target_torque2   = 0.0f;
MessageType current_mode2    = CONTROL_TORQUE;

// ── CAN encoder feedback timing ───────────────────────────────────────────────
static unsigned long lastFeedbackUs = 0;
const  unsigned long FEEDBACK_US    = 10000;  // 100 Hz

// ─────────────────────────────────────────────────────────────────────────────
// STOP MOTORS — zero torque
// ─────────────────────────────────────────────────────────────────────────────
void stopMotors() {
  motor.target  = 0.0f;
  motor2.target = 0.0f;
  Serial.println(F("[BAL] DISABLED — motors zeroed"));
  bluetooth.println(F("[BAL] DISABLED — motors zeroed"));
}

void enableBalancing() {
  motor.target  = 0.0f;
  motor2.target = 0.0f;
  vel_integral  = 0.0f;
  start_shaft_angle1 = motor.shaft_angle;
  start_shaft_angle2 = motor2.shaft_angle;
  current_pos_x      = 0.0f;
  balancing_enabled = true;
  Serial.println(F("[BAL] ENABLED — RL balancing active"));
  bluetooth.println(F("[BAL] ENABLED — RL balancing active"));
}

// ─────────────────────────────────────────────────────────────────────────────
// STATUS & TELEMETRY (Serial Monitor or Bluetooth terminal)
//   E       → enable balancing
//   D       → disable balancing
//   P<val>  → set sim_pitch_eq (default 0.310 rad)
//   T<val>  → set pitch_trim (radians, e.g. T0.02 or T-0.02)
//   S<val>  → set torque_scale (e.g. S0.5 or S0.7)
//   G       → toggle gyro direction (+1 / -1)
//   ?       → print live status, sensors & neural network outputs
// ─────────────────────────────────────────────────────────────────────────────
void printStatus(Stream& port) {
  float current_pitch = getPitchIMU();
  float real_tilt = current_pitch - pitch_trim;
  float policy_pitch_in = real_tilt + sim_pitch_eq + last_vel_trim;

  port.println(F("=================================================="));
  port.printf("RL Policy:   %d layers, obs_dim=%d, action_dim=%d\n",
              ACTOR_NUM_LAYERS, ACTOR_OBS_SIZE, ACTOR_ACTION_SIZE);
  port.printf("Config:      scale=%.2f  filter=%.2f  trim=%.4f rad (%.2f deg)  motor_sign=%.1f  gyro_sign=%.1f\n",
              torque_scale, action_lpf, pitch_trim, pitch_trim * 57.2958f, motor_sign, gyro_sign);
  port.printf("Real Tilt:   %.3f rad (%.1f deg)  [0.0 = vertical]\n",
              real_tilt, real_tilt * 57.2958f);
  port.printf("Policy In:   obs[0]=%.3f rad (should be ~%.3f when balanced!)\n",
              policy_pitch_in, sim_pitch_eq);
  port.printf("Gyro Rate:   obs[1]=%.3f rad/s (positive when falling forward)\n",
              last_pitch_rate);
  port.printf("Position:    x=%.3f m (%.1f cm) [from start point]\n",
              current_pos_x, current_pos_x * 100.0f);
  port.printf("Velocity:    v=%.2f rad/s  trim=%.3f rad  vel_i=%.3f (gain_p=%.4f, gain_i=%.4f)\n",
              last_avg_vel, last_vel_trim, vel_integral, vel_gain_p, vel_gain_i);
  port.printf("Encoders:    L=%.2f rad/s  R=%.2f rad/s\n",
              motor.shaft_velocity, motor2.shaft_velocity);
  port.printf("Actor NN:    act[0]=%.3f  act[1]=%.3f\n",
              last_action_left, last_action_right);
  port.printf("Motor V:     M1=%.2f V   M2=%.2f V\n",
              motor.target, motor2.target);
  if (motor.target > 0.1f) {
    port.println(F("Wheels Spin: FORWARD (Driving forward to catch fall!) [CORRECT]"));
  } else if (motor.target < -0.1f) {
    port.println(F("Wheels Spin: BACKWARD (Send 'M' to invert motor direction!)"));
  } else {
    port.println(F("Wheels Spin: IDLE / NEAR ZERO"));
  }
  port.println(F("=================================================="));
}

void calibratePitchZero(Stream& port) {
  port.println(F("=================================================="));
  port.println(F("[CAL] Hold robot upright at perfect balance point..."));
  port.println(F("[CAL] Sampling 100 IMU readings (~2 seconds)..."));

  // Temporarily zero motors while holding
  motor.target  = 0.0f;
  motor2.target = 0.0f;
  motor.move();
  motor2.move();

  float sum_pitch = 0.0f;
  int count = 0;
  unsigned long timeout = millis() + 4000;

  while (count < 100 && millis() < timeout) {
    loopIMU();
    if (cache_valid) {
      cache_valid = false;
      sum_pitch += getPitchIMU();
      count++;
      if (count % 25 == 0) {
        port.printf("[CAL] Progress: %d / 100\n", count);
      }
    }
    delay(5);
  }

  if (count >= 100) {
    pitch_trim = sum_pitch / (float)count;
    vel_integral = 0.0f;

    // Save to Flash (Preferences)
    Preferences prefs;
    if (prefs.begin("balance", false)) {
      prefs.putFloat("trim", pitch_trim);
      prefs.end();
      port.println(F("[CAL] Saved pitch zero offset to Flash memory!"));
    }

    port.printf("[CAL] SUCCESS! New pitch_trim = %.4f rad (%.2f deg)\n",
                pitch_trim, pitch_trim * 57.2958f);
    port.println(F("[CAL] Balance reference set! Robot will balance at this angle."));
  } else {
    port.println(F("[CAL] WARNING: IMU read timeout, calibration incomplete!"));
  }
  port.println(F("=================================================="));
}

void handleTuning(Stream& port) {
  if (!port.available()) return;
  String line = port.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  if (line.equalsIgnoreCase("E")) { enableBalancing();                       return; }
  if (line.equalsIgnoreCase("D")) { balancing_enabled = false; stopMotors(); return; }
  if (line[0] == '?')            { printStatus(Serial); printStatus(bluetooth); return; }
  if (line.equalsIgnoreCase("C")) { calibratePitchZero(port);                 return; }

  if (line.equalsIgnoreCase("M")) {
    motor_sign = -motor_sign;
    port.printf("[MOTOR] Toggled motor_sign = %.1f\n", motor_sign);
    return;
  }
  if (line.equalsIgnoreCase("G")) {
    gyro_sign = -gyro_sign;
    port.printf("[GYRO] Toggled gyro_sign = %.1f\n", gyro_sign);
    return;
  }
  if (line.equalsIgnoreCase("Z")) {
    vel_integral = 0.0f;
    start_shaft_angle1 = motor.shaft_angle;
    start_shaft_angle2 = motor2.shaft_angle;
    current_pos_x = 0.0f;
    port.println(F("[ORIGIN] Reset origin (x=0.0m) to current spot"));
    return;
  }

  char cmd = toupper(line[0]);
  float val = line.substring(1).toFloat();

  if (cmd == 'V') {
    vel_gain_i = constrain(val, 0.0f, 1.0f);
    vel_integral = 0.0f;
    port.printf("[VEL_I] Set vel_gain_i = %.4f (position hold stiffness, 0=off)\n", vel_gain_i);
  } else if (cmd == 'B') {
    vel_gain_p = constrain(val, 0.0f, 0.2f);
    port.printf("[VEL_P] Set vel_gain_p = %.4f (velocity damping brake gain)\n", vel_gain_p);
  } else if (cmd == 'P') {
    sim_pitch_eq = val;
    port.printf("[PITCH_EQ] Set sim_pitch_eq = %.4f rad (%.2f deg)\n", sim_pitch_eq, sim_pitch_eq * 57.2958f);
  } else if (cmd == 'T') {
    pitch_trim = val;
    port.printf("[TRIM] Set pitch_trim = %.4f rad (%.2f deg)\n", pitch_trim, pitch_trim * 57.2958f);
  } else if (cmd == 'S') {
    torque_scale = constrain(val, 0.1f, 2.5f);
    port.printf("[SCALE] Set torque_scale = %.2f\n", torque_scale);
  } else if (cmd == 'F') {
    action_lpf = constrain(val, 0.1f, 1.0f);
    port.printf("[FILTER] Set action_lpf = %.2f (lower=smoother, 1.0=raw)\n", action_lpf);
  } else {
    port.println(F("Commands: '?'=status, 'C'=calibrate zero offset, 'V<gain>'=stiffness, 'B<gain>'=brake, 'Z'=zero int, 'S<scale>'=gain, 'T<rad>'=trim"));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(250000);
  bluetooth.begin("FOCBalancer");
  _delay(1000);

  if (!initIMU()) {
    Serial.println(F("IMU error - halting!"));
    while (true);
  }
  _delay(1000);

  initCAN();

  encoder1.init();
  encoder2.init();
  motor.linkSensor(&encoder1);
  motor2.linkSensor(&encoder2);

  driver1.voltage_power_supply = 8;
  driver1.init();
  motor.linkDriver(&driver1);

  driver2.voltage_power_supply = 8;
  driver2.init();
  motor2.linkDriver(&driver2);

  motor.controller  = MotionControlType::torque;
  motor2.controller = MotionControlType::torque;

  motor.useMonitoring(Serial);
  motor2.useMonitoring(Serial);

  motor.init();
  motor2.init();
  motor.initFOC();
  motor2.initFOC();

  // Load saved pitch zero offset from Flash memory (calibrated via 'C' command)
  Preferences prefs;
  if (prefs.begin("balance", true)) {
    pitch_trim = prefs.getFloat("trim", 0.0f);
    prefs.end();
    Serial.printf("[INIT] Loaded saved pitch zero offset: %.4f rad (%.2f deg)\n",
                  pitch_trim, pitch_trim * 57.2958f);
    bluetooth.printf("[INIT] Loaded saved pitch zero offset: %.4f rad (%.2f deg)\n",
                     pitch_trim, pitch_trim * 57.2958f);
  }

  Serial.println(F("Ready — RL Balance Bot! Hold upright & send 'C' anytime to calibrate zero offset."));
  Serial.println(F("Commands: '?'=status, 'C'=calibrate zero, 'M'=motor dir, 'G'=gyro dir, 'S<scale>'=power"));
  bluetooth.println(F("Ready — RL Balance Bot! Hold upright & send 'C' anytime to calibrate zero offset."));
  bluetooth.println(F("Commands: '?'=status, 'C'=calibrate zero, 'M'=motor dir, 'G'=gyro dir, 'S<scale>'=power"));
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN LOOP
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
  // ── FOC — must run as fast as possible ───────────────────────────────────
  motor.loopFOC();
  motor2.loopFOC();

  motor.move();
  motor2.move();

  // ── CAN receive — handles enable + cmd_vel + motor commands ──────────────
  receiveCANMessages();

  // ── IMU — update cache every loop ────────────────────────────────────────
  loopIMU();

  if (cache_valid) {
    cache_valid = false;

    // Always send IMU so telemetry / dashboard can monitor angle
    sendIMUData();

    if (!balancing_enabled) {
      motor.target  = 0.0f;
      motor2.target = 0.0f;
    } else {
      float current_pitch = getPitchIMU();

      // Read instantaneous physical gyroscope from MPU6050 with gentle smoothing
      float raw_pitch_rate = getPitchRateIMU() * gyro_sign;
      static float filtered_pitch_rate = 0.0f;
      filtered_pitch_rate = 0.75f * raw_pitch_rate + 0.25f * filtered_pitch_rate;
      last_pitch_rate = filtered_pitch_rate;

      // Real robot tilt relative to upright
      float real_tilt = current_pitch - pitch_trim;

      // ── Safety Fall Cutoff ───────────────────────────────────────────────
      // If robot has fallen over (> 34.4 deg), kill motors for safety
      if (fabs(real_tilt) > MAX_FALL_ANGLE) {
        motor.target  = 0.0f;
        motor2.target = 0.0f;
        vel_integral  = 0.0f;
      } else {
        // ── 1. Average Wheel Velocity & Zero-Drift Velocity Regulation ─────
        // Compute forward velocity of robot in rad/s
        float avg_vel = 0.5f * (motor.shaft_velocity + motor2.shaft_velocity) * motor_sign;
        last_avg_vel = avg_vel;

        // Velocity setpoint from CAN / joystick cmd_vel_linear (if present)
        #define CMD_VEL_LINEAR_SCALE 15.0f
        float vel_setpoint = cmd_vel_linear * CMD_VEL_LINEAR_SCALE;
        float vel_error = avg_vel - vel_setpoint;

        // Anti-windup position / distance accumulation:
        // dt = 0.02s (at 50 Hz DMP rate). Clamped to +/- 10.0 rad of wheel angle (~33 cm travel).
        vel_integral = constrain(vel_integral + vel_error * 0.02f, -10.0f, 10.0f);

        // Active holding lean setpoint:
        // Clamped to +/- 0.08 rad (~4.5 degrees) — enough to overcome friction without destabilizing balance!
        float raw_trim = constrain((vel_error * vel_gain_p) + (vel_integral * vel_gain_i), -0.08f, 0.08f);

        // Low-pass filter (0.50 new, 0.50 prev): fast response to eliminate phase lag oscillation
        static float filtered_trim = 0.0f;
        filtered_trim = 0.50f * raw_trim + 0.50f * filtered_trim;
        last_vel_trim = filtered_trim;

        // ── 2. Construct Actor Observation Vector ───────────────────────────
        #if ACTOR_OBS_SIZE == 5
        // ── 5-Observation Policy (Direct Neural Network Stay-in-Place Centering) ─
        // [0] pitch: maps real vertical to policy equilibrium (sim_pitch_eq)
        // [1] pitch_rate: filtered hardware gyro angular velocity (rad/s)
        // [2] left_wheel_vel * 0.1 (rad/s)
        // [3] right_wheel_vel * 0.1 (rad/s)
        // [4] x_pos: displacement from starting spot (meters)
        float d_angle1 = (motor.shaft_angle - start_shaft_angle1) * motor_sign;
        float d_angle2 = (motor2.shaft_angle - start_shaft_angle2) * motor_sign;
        current_pos_x = 0.5f * (d_angle1 + d_angle2) * WHEEL_RADIUS;

        float obs[5];
        obs[0] = real_tilt + sim_pitch_eq;
        obs[1] = filtered_pitch_rate;
        obs[2] = motor.shaft_velocity * motor_sign * 0.1f;
        obs[3] = motor2.shaft_velocity * motor_sign * 0.1f;
        obs[4] = current_pos_x;
        #else
        // ── 4-Observation Policy (Standard Balancing) ────────────────────────
        float obs[4];
        obs[0] = real_tilt + sim_pitch_eq + filtered_trim;
        obs[1] = filtered_pitch_rate;
        obs[2] = motor.shaft_velocity * motor_sign * 0.1f;
        obs[3] = motor2.shaft_velocity * motor_sign * 0.1f;
        #endif

        // ── 3. Run Pure C Neural Network Inference (< 25 microseconds) ─────
        float raw_action[2] = {0.0f, 0.0f};
        actor_forward(obs, raw_action);

        // ── 4. Smooth Action Output (removes high-frequency jitter / vibration) ─
        static float filtered_action[2] = {0.0f, 0.0f};
        filtered_action[0] = action_lpf * raw_action[0] + (1.0f - action_lpf) * filtered_action[0];
        filtered_action[1] = action_lpf * raw_action[1] + (1.0f - action_lpf) * filtered_action[1];

        last_action_left  = filtered_action[0];
        last_action_right = filtered_action[1];

        // ── 5. Turning difference from joystick / CAN cmd_vel_angular ───────
        float turn = cmd_vel_angular * CMD_VEL_ANGULAR_SCALE;

        // ── 6. Set Motor Target Voltages ────────────────────────────────────
        motor.target  = constrain((filtered_action[0] * motor_sign * MAX_VOLTAGE * torque_scale) + turn, -MAX_VOLTAGE, MAX_VOLTAGE);
        motor2.target = constrain((filtered_action[1] * motor_sign * MAX_VOLTAGE * torque_scale) - turn, -MAX_VOLTAGE, MAX_VOLTAGE);
      }
    }
  }

  // ── CAN encoder feedback at 100 Hz ───────────────────────────────────────
  unsigned long now = micros();
  if (now - lastFeedbackUs >= FEEDBACK_US) {
    lastFeedbackUs = now;
    sendFeedback();
    sendFeedback2();
  }

  // ── Serial / Bluetooth tuning & telemetry ─────────────────────────────────
  handleTuning(Serial);
  handleTuning(bluetooth);
}
