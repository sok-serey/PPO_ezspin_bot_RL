#include "can.h"
#include <Arduino.h>
#include <SimpleFOC.h>
#include "imu.h"

#define CAN_RX_PIN  4
#define CAN_TX_PIN  5

extern BLDCMotor motor;
extern BLDCMotor motor2;
extern float target_position,  target_velocity,  target_torque;
extern float target_position2, target_velocity2, target_torque2;
extern MessageType current_mode, current_mode2;

// ── Runtime configurable CAN IDs & baud ──────────────────────────────────────
uint32_t can_id_motor1_cmd = 1;
uint32_t can_id_motor2_cmd = 2;
uint32_t can_id_motor1_fb  = 0x101;
uint32_t can_id_motor2_fb  = 0x102;
uint32_t can_id_imu        = 200;
uint32_t can_baudrate      = 1000000;

unsigned long lastCANMessageTime  = 0;
unsigned long lastCANMessageTime2 = 0;

// ─────────────────────────────────────────────────────────────────────────────
static bool getBaudConfig(uint32_t baud, twai_timing_config_t* t_config) {
  switch (baud) {
    case 25000:   *t_config = TWAI_TIMING_CONFIG_25KBITS();   return true;
    case 50000:   *t_config = TWAI_TIMING_CONFIG_50KBITS();   return true;
    case 100000:  *t_config = TWAI_TIMING_CONFIG_100KBITS();  return true;
    case 125000:  *t_config = TWAI_TIMING_CONFIG_125KBITS();  return true;
    case 250000:  *t_config = TWAI_TIMING_CONFIG_250KBITS();  return true;
    case 500000:  *t_config = TWAI_TIMING_CONFIG_500KBITS();  return true;
    case 800000:  *t_config = TWAI_TIMING_CONFIG_800KBITS();  return true;
    case 1000000: *t_config = TWAI_TIMING_CONFIG_1MBITS();    return true;
    default:      return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void initCAN() {
  Serial.printf("Starting ESP32 TWAI/CAN @ %lu bps...\n", can_baudrate);

  twai_general_config_t g_config = TWAI_GENERAL_CONFIG_DEFAULT(
    (gpio_num_t)CAN_TX_PIN,
    (gpio_num_t)CAN_RX_PIN,
    TWAI_MODE_NORMAL
  );

  twai_timing_config_t t_config;
  if (!getBaudConfig(can_baudrate, &t_config)) {
    Serial.printf("Invalid baudrate %lu, defaulting to 1Mbps\n", can_baudrate);
    t_config = TWAI_TIMING_CONFIG_1MBITS();
    can_baudrate = 1000000;
  }

  twai_filter_config_t f_config = TWAI_FILTER_CONFIG_ACCEPT_ALL();

  if (twai_driver_install(&g_config, &t_config, &f_config) != ESP_OK) {
    Serial.println("TWAI driver install failed!");
    while (1) delay(1000);
  }
  if (twai_start() != ESP_OK) {
    Serial.println("TWAI start failed!");
    while (1) delay(1000);
  }

  Serial.printf(
    "CAN ready — M1_CMD=%lu M2_CMD=%lu | M1_FB=0x%03lX M2_FB=0x%03lX | IMU=%lu | ENABLE=0x%03X | CMD_VEL=0x%03X\n",
    can_id_motor1_cmd, can_id_motor2_cmd,
    can_id_motor1_fb,  can_id_motor2_fb,
    can_id_imu, CAN_ID_ENABLE, CAN_ID_CMD_VEL
  );
}

// ─────────────────────────────────────────────────────────────────────────────
void restartCAN() {
  twai_stop();
  twai_driver_uninstall();
  delay(100);
  initCAN();
}

// ─────────────────────────────────────────────────────────────────────────────
// receiveCANMessages — single non-blocking receive point for ALL frame types
//
// Frame types handled:
//   CAN_ID_ENABLE  (0x010) — 1 byte  — enable/disable balancing
//   CAN_ID_CMD_VEL (0x020) — 8 bytes — joystick linear.x + angular.z
//   can_id_motor1_cmd (1)  — 5 bytes — motor1 torque/speed/position
//   can_id_motor2_cmd (2)  — 5 bytes — motor2 torque/speed/position
//
// Drains up to 5 frames per call — non-blocking (timeout=0)
// ─────────────────────────────────────────────────────────────────────────────
void receiveCANMessages() {
  twai_message_t rxMsg;

  for (int i = 0; i < 5; i++) {
    if (twai_receive(&rxMsg, 0) != ESP_OK) break;

    // ── Enable / disable ──────────────────────────────────────────────────
    if (rxMsg.identifier == CAN_ID_ENABLE && rxMsg.data_length_code >= 1) {
      if (rxMsg.data[0] == 0x01 && !balancing_enabled) {
        enableBalancing();
      } else if (rxMsg.data[0] == 0x00 && balancing_enabled) {
        balancing_enabled = false;
        stopMotors();
      }
      continue;
    }

    // ── cmd_vel from joystick ─────────────────────────────────────────────
    // data[0..3] = linear.x  (float) — forward/back m/s
    // data[4..7] = angular.z (float) — turn rad/s
    if (rxMsg.identifier == CAN_ID_CMD_VEL && rxMsg.data_length_code == 8) {
      memcpy((void*)&cmd_vel_linear,  &rxMsg.data[0], sizeof(float));
      memcpy((void*)&cmd_vel_angular, &rxMsg.data[4], sizeof(float));
      continue;
    }

    // ── Motor commands — must be exactly 5 bytes ──────────────────────────
    if (rxMsg.data_length_code != 5) continue;

    uint8_t received_mode = rxMsg.data[0];
    float   new_target;
    memcpy(&new_target, &rxMsg.data[1], sizeof(float));

    // Motor 1
    if (rxMsg.identifier == can_id_motor1_cmd) {
      lastCANMessageTime = millis();
      if (received_mode == CONTROL_POSITION && current_mode != CONTROL_POSITION) {
        motor.controller = MotionControlType::angle;
        current_mode = CONTROL_POSITION; target_position = new_target;
      } else if (received_mode == CONTROL_SPEED && current_mode != CONTROL_SPEED) {
        motor.controller = MotionControlType::velocity;
        current_mode = CONTROL_SPEED; target_velocity = new_target;
      } else if (received_mode == CONTROL_TORQUE && current_mode != CONTROL_TORQUE) {
        motor.controller = MotionControlType::torque;
        current_mode = CONTROL_TORQUE; target_torque = new_target;
      } else {
        float& t = (current_mode == CONTROL_POSITION) ? target_position :
                   (current_mode == CONTROL_SPEED)    ? target_velocity : target_torque;
        if (fabs(new_target - t) > 0.05f) t = new_target;
      }
    }

    // Motor 2
    else if (rxMsg.identifier == can_id_motor2_cmd) {
      lastCANMessageTime2 = millis();
      if (received_mode == CONTROL_POSITION && current_mode2 != CONTROL_POSITION) {
        motor2.controller = MotionControlType::angle;
        current_mode2 = CONTROL_POSITION; target_position2 = new_target;
      } else if (received_mode == CONTROL_SPEED && current_mode2 != CONTROL_SPEED) {
        motor2.controller = MotionControlType::velocity;
        current_mode2 = CONTROL_SPEED; target_velocity2 = new_target;
      } else if (received_mode == CONTROL_TORQUE && current_mode2 != CONTROL_TORQUE) {
        motor2.controller = MotionControlType::torque;
        current_mode2 = CONTROL_TORQUE; target_torque2 = new_target;
      } else {
        float& t = (current_mode2 == CONTROL_POSITION) ? target_position2 :
                   (current_mode2 == CONTROL_SPEED)    ? target_velocity2 : target_torque2;
        if (fabs(new_target - t) > 0.05f) t = new_target;
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// sendFeedback / sendFeedback2 — rate controlled by .ino (100 Hz)
// ─────────────────────────────────────────────────────────────────────────────
void sendFeedback() {
  twai_message_t txMsg;
  txMsg.identifier = can_id_motor1_fb; txMsg.extd = 0;
  txMsg.rtr = 0; txMsg.data_length_code = 8;
  float a = motor.shaft_angle,  v = motor.shaft_velocity;
  memcpy(&txMsg.data[0], &a, sizeof(float));
  memcpy(&txMsg.data[4], &v, sizeof(float));
  twai_transmit(&txMsg, 0);
}

void sendFeedback2() {
  twai_message_t txMsg;
  txMsg.identifier = can_id_motor2_fb; txMsg.extd = 0;
  txMsg.rtr = 0; txMsg.data_length_code = 8;
  float a = motor2.shaft_angle, v = motor2.shaft_velocity;
  memcpy(&txMsg.data[0], &a, sizeof(float));
  memcpy(&txMsg.data[4], &v, sizeof(float));
  twai_transmit(&txMsg, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// sendIMUData — called from .ino on every IMU packet (~100 Hz)
// ─────────────────────────────────────────────────────────────────────────────
void sendIMUData() {
  twai_message_t txMsg;
  txMsg.identifier = can_id_imu; txMsg.extd = 0;
  txMsg.rtr = 0; txMsg.data_length_code = 8;
  float r = getRollIMU(), p = getPitchIMU();
  memcpy(&txMsg.data[0], &r, sizeof(float));
  memcpy(&txMsg.data[4], &p, sizeof(float));
  twai_transmit(&txMsg, 0);
}