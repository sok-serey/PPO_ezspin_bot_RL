#ifndef CAN_H
#define CAN_H

#include <ESP32-TWAI-CAN.hpp>
#include <SimpleFOC.h>

enum MessageType {
  CONTROL_TORQUE   = 0x01,
  CONTROL_SPEED    = 0x02,
  CONTROL_POSITION = 0x03
};

// ── CAN IDs ───────────────────────────────────────────────────────────────────
extern uint32_t can_id_motor1_cmd;   // default 1
extern uint32_t can_id_motor2_cmd;   // default 2
extern uint32_t can_id_motor1_fb;    // default 0x101
extern uint32_t can_id_motor2_fb;    // default 0x102
extern uint32_t can_id_imu;          // default 200
extern uint32_t can_baudrate;        // default 1000000

#define CAN_ID_ENABLE   0x010   // enable/disable balancing
                                //   data[0] = 0x01 → enable
                                //   data[0] = 0x00 → disable

#define CAN_ID_CMD_VEL  0x020   // joystick velocity command
                                //   data[0..3] = linear.x  (float, m/s)
                                //   data[4..7] = angular.z (float, rad/s)

// ── balancing_enabled — defined in .ino ──────────────────────────────────────
extern volatile bool balancing_enabled;

// Callbacks defined in .ino — called by receiveCANMessages()
extern void enableBalancing();
extern void stopMotors();

// ── cmd_vel from joystick — defined in .ino ───────────────────────────────────
extern volatile float cmd_vel_linear;   // linear.x  — forward/back setpoint
extern volatile float cmd_vel_angular;  // angular.z — left/right turn

// ── Motor targets ─────────────────────────────────────────────────────────────
extern float target_position;
extern float target_velocity;
extern float target_torque;
extern MessageType current_mode;

extern float target_position2;
extern float target_velocity2;
extern float target_torque2;
extern MessageType current_mode2;

extern unsigned long lastCANMessageTime;
extern unsigned long lastCANMessageTime2;

// ── Functions ─────────────────────────────────────────────────────────────────
void initCAN();
void restartCAN();
void receiveCANMessages();
void sendFeedback();
void sendFeedback2();
void sendIMUData();

#endif