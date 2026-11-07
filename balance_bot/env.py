"""Gymnasium environment for a two-wheel self-balancing robot in MuJoCo."""

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

MODEL_PATH = Path(__file__).parent / "assets" / "balance_bot.xml"

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 1.2,
    "elevation": -15.0,
    "azimuth": 90.0,
}


class BalanceBotEnv(MujocoEnv):
    """Keep the robot balanced and (optionally) track a forward velocity command.

    The CAD centre of mass is behind the wheel axle, so the robot balances at a
    lean (``pitch_eq``, ~0.31 rad) rather than at pitch = 0. Observations report
    the raw IMU pitch, like the real sensor; reward and termination use the
    error from the balance angle.

    Observation modes:
      - 'robot' (default, 4 dimensions):
          [0] pitch (rad) - IMU pitch angle (with calibration bias and noise)
          [1] pitch_rate (rad/s) - Gyro Y pitch angular velocity (with bias and noise)
          [2] left_wheel_vel * 0.1 (rad/s) - Left wheel speed
          [3] right_wheel_vel * 0.1 (rad/s) - Right wheel speed
          ([4] cmd_forward_vel if command_vel=True)

      - 'full' (legacy, 10 dimensions):
          pitch, roll, pitch_rate, roll_rate, yaw_rate,
          left_wheel_vel, right_wheel_vel,
          cmd_forward_vel, prev_action_left, prev_action_right

    Action (2): normalized wheel torques in [-1, 1].
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        frame_skip: int = 5,  # 0.002 s * 5 = 100 Hz control
        max_init_pitch: float = 0.15,  # around the balance angle
        fall_angle: float = 0.6,  # from the balance angle
        command_vel: bool = False,
        obs_mode: str = "robot_pos",  # "robot_pos" (5 obs), "robot" (4 obs) or "full" (10 obs)
        obs_noise: float = 0.02,
        alive_bonus: float = 1.0,
        pitch_penalty_coef: float = 5.0,
        pitch_rate_penalty_coef: float = 0.05,
        action_penalty_coef: float = 0.01,
        position_penalty_coef: float = 1.5,
        yaw_penalty_coef: float = 0.1,
        max_pos_distance: float = 0.8,
        randomize_dynamics: bool = True,
        push_disturbances: bool = True,
        render_mode: str | None = None,
        **kwargs,
    ):
        self.max_init_pitch = max_init_pitch
        self.fall_angle = fall_angle
        self.command_vel = command_vel
        self.obs_mode = obs_mode
        self.obs_noise = obs_noise

        # Reward coefficients
        self.alive_bonus = alive_bonus
        self.pitch_penalty_coef = pitch_penalty_coef
        self.pitch_rate_penalty_coef = pitch_rate_penalty_coef
        self.action_penalty_coef = action_penalty_coef
        self.position_penalty_coef = position_penalty_coef
        self.yaw_penalty_coef = yaw_penalty_coef
        self.max_pos_distance = max_pos_distance

        self.randomize_dynamics = randomize_dynamics
        self.push_disturbances = push_disturbances
        self.cmd_vel = 0.0
        self.prev_action = np.zeros(2)
        self.prev_applied_action = np.zeros(2)

        # Domain randomization state
        self.motor_scales = np.ones(2)
        self.motor_deadband = 0.0
        self.action_lag = 0.0
        self.pitch_bias = 0.0
        self.gyro_bias = 0.0
        self.pos_bias = 0.0
        self.wheel_vel_bias = np.zeros(2)
        self.disturbance_steps = 0
        self.disturbance_force = 0.0

        if self.obs_mode == "robot":
            obs_dim = 5 if self.command_vel else 4
        elif self.obs_mode == "robot_pos":
            obs_dim = 6 if self.command_vel else 5
        else:
            obs_dim = 10
        observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64)

        super().__init__(
            str(MODEL_PATH),
            frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        self._nominal_mass = self.model.body_mass.copy()
        self._nominal_friction = self.model.geom_friction.copy()
        self._chassis_id = self.model.body("chassis").id
        self._nominal_chassis_com = self.model.body_ipos[self._chassis_id].copy()
        self._wheel_radius = float(self.model.geom("wheel_left").size[0])
        # Axle centre in the chassis frame (wheels share x/z, differ in y).
        self._axle = self.model.body("left_wheel_link").pos.copy()
        self._axle[1] = 0.0
        self._wheel_dof_indices = [
            self.model.joint("left_wheel_joint").dofadr[0],
            self.model.joint("right_wheel_joint").dofadr[0],
        ]
        self._nominal_damping = self.model.dof_damping.copy()
        self._nominal_frictionloss = self.model.dof_frictionloss.copy()
        self.pitch_eq = self._balance_pitch()

    # ------------------------------------------------------------------ helpers
    def _sensor(self, name: str) -> np.ndarray:
        return self.data.sensor(name).data

    def _pitch_roll(self) -> tuple[float, float]:
        w, x, y, z = self._sensor("imu_quat")
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return float(pitch), float(roll)

    def _balance_pitch(self) -> float:
        """Pitch at which the whole-robot centre of mass is directly above the axle."""
        com, mass = np.zeros(3), 0.0
        for body in ("chassis", "left_wheel_link", "right_wheel_link"):
            b = self.model.body(body)
            offset = b.pos if body != "chassis" else 0.0
            com += b.mass[0] * (offset + b.ipos)
            mass += b.mass[0]
        dx, _, dz = com / mass - self._axle
        return float(np.arctan2(-dx, dz))

    def _forward_vel(self) -> float:
        # Average wheel speed * radius, i.e. what wheel encoders would report.
        wl, wr = self._sensor("wheel_left_vel")[0], self._sensor("wheel_right_vel")[0]
        return 0.5 * (wl + wr) * self._wheel_radius

    def _get_obs(self) -> np.ndarray:
        pitch, roll = self._pitch_roll()
        gyro = self._sensor("imu_gyro")  # in IMU frame: x=roll, y=pitch, z=yaw rates
        wl = self._sensor("wheel_left_vel")[0]
        wr = self._sensor("wheel_right_vel")[0]

        if self.obs_mode in ["robot", "robot_pos"]:
            # Observations available on real robot:
            # 0: pitch (rad)
            # 1: pitch_rate (rad/s)
            # 2: left_wheel_vel * 0.1
            # 3: right_wheel_vel * 0.1
            # [4: x_pos (m) if robot_pos mode]
            obs_list = [
                pitch + self.pitch_bias,
                gyro[1] + self.gyro_bias,
                (wl + self.wheel_vel_bias[0]) * 0.1,
                (wr + self.wheel_vel_bias[1]) * 0.1,
            ]
            noise_list = [
                self.np_random.normal(0.0, self.obs_noise),
                self.np_random.normal(0.0, self.obs_noise * 1.5),
                self.np_random.normal(0.0, self.obs_noise * 0.5),
                self.np_random.normal(0.0, self.obs_noise * 0.5),
            ]
            if self.obs_mode == "robot_pos":
                x_pos = float(self.data.qpos[0])
                obs_list.append(x_pos + self.pos_bias)
                noise_list.append(self.np_random.normal(0.0, self.obs_noise * 0.5))

            obs = np.array(obs_list, dtype=np.float64)
            if self.obs_noise > 0.0:
                obs += np.array(noise_list, dtype=np.float64)
            if self.command_vel:
                obs = np.append(obs, self.cmd_vel)
            return obs

        else:
            # Full 10-observation mode (backward compatibility)
            obs = np.array(
                [
                    pitch + self.pitch_bias,
                    roll,
                    gyro[1] + self.gyro_bias,
                    gyro[0],
                    gyro[2],
                    (wl + self.wheel_vel_bias[0]) * 0.1,
                    (wr + self.wheel_vel_bias[1]) * 0.1,
                    self.cmd_vel,
                    *self.prev_action,
                ],
                dtype=np.float64,
            )
            if self.obs_noise > 0.0:
                obs[:7] += self.np_random.normal(0.0, self.obs_noise, size=7)
            return obs

    # --------------------------------------------------------------------- gym
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Domain Randomization on Actuators:
        # 1. Asymmetric motor torque scaling (simulates motor variation & battery level)
        action_scaled = action * self.motor_scales

        # 2. Static friction / motor deadband
        if self.motor_deadband > 0.0:
            mask = np.abs(action_scaled) < self.motor_deadband
            action_scaled[mask] = 0.0

        # 3. First-order actuator lag (simulates motor inductance & communication latency)
        applied_action = (1.0 - self.action_lag) * action_scaled + self.action_lag * self.prev_applied_action
        self.prev_applied_action = applied_action.copy()

        # Domain Randomization on Disturbances:
        # Occasional random horizontal push impulse to the chassis COM
        if self.randomize_dynamics and self.push_disturbances:
            if self.disturbance_steps > 0:
                self.data.xfrc_applied[self._chassis_id, 0] = self.disturbance_force
                self.disturbance_steps -= 1
            else:
                self.data.xfrc_applied[self._chassis_id, :] = 0.0
                if self.np_random.random() < 0.015:  # ~1.5% chance per step (~every 0.7s)
                    self.disturbance_force = float(
                        self.np_random.choice([-1.0, 1.0]) * self.np_random.uniform(1.0, 2.5)
                    )
                    self.disturbance_steps = int(self.np_random.integers(3, 8))

        self.do_simulation(applied_action, self.frame_skip)
        self.data.xfrc_applied[self._chassis_id, :] = 0.0

        pitch, roll = self._pitch_roll()
        gyro = self._sensor("imu_gyro")
        fwd_vel = self._forward_vel()

        pitch_err = pitch - self.pitch_eq
        vel_err = fwd_vel - self.cmd_vel

        # Reward structure matching user specification:
        #   + alive_bonus for keeping upright
        #   - pitch^2 penalty (deviation from equilibrium angle)
        #   - pitch_rate^2 penalty (gyro damping against oscillation)
        #   - action^2 penalty (discourages high motor torque / heating)
        #   - position penalty (drift from origin)
        #   - yaw penalty (rotation around Z axis)
        pitch_penalty = self.pitch_penalty_coef * (pitch_err**2)
        pitch_rate_penalty = self.pitch_rate_penalty_coef * (gyro[1]**2)
        action_penalty = self.action_penalty_coef * float(np.sum(action**2))

        x_pos = float(self.data.qpos[0])
        y_pos = float(self.data.qpos[1])
        pos_dist = float(np.sqrt(x_pos**2 + y_pos**2))
        position_penalty = self.position_penalty_coef * (x_pos**2 + y_pos**2) + (0.5 * self.position_penalty_coef) * pos_dist

        yaw_rate = float(gyro[2])
        yaw_penalty = self.yaw_penalty_coef * abs(yaw_rate)

        vel_penalty = 0.2 * (fwd_vel**2) if not self.command_vel else 0.0

        reward = (
            self.alive_bonus
            - pitch_penalty
            - pitch_rate_penalty
            - action_penalty
            - position_penalty
            - yaw_penalty
            - vel_penalty
        )
        self.prev_action = action.copy()

        out_of_bounds = bool(abs(x_pos) > self.max_pos_distance) if self.obs_mode == "robot_pos" else False
        terminated = bool(abs(pitch_err) > self.fall_angle or abs(roll) > self.fall_angle or out_of_bounds)
        if terminated:
            reward -= 10.0

        if self.render_mode == "human":
            self.render()

        info = {"pitch": pitch, "pitch_eq": self.pitch_eq, "forward_vel": fwd_vel, "cmd_vel": self.cmd_vel}
        return self._get_obs(), reward, terminated, False, info

    def reset_model(self) -> np.ndarray:
        if self.randomize_dynamics:
            # 1. Physics domain randomization
            self.model.body_mass[:] = self._nominal_mass * self.np_random.uniform(0.8, 1.2, size=self._nominal_mass.shape)
            self.model.geom_friction[:, 0] = self._nominal_friction[:, 0] * self.np_random.uniform(0.6, 1.4)
            # CAD centre of mass is never exact: shift it by up to 6 mm in x/z.
            com = self._nominal_chassis_com.copy()
            com[[0, 2]] += self.np_random.uniform(-0.006, 0.006, size=2)
            self.model.body_ipos[self._chassis_id] = com

            # Wheel joint damping and friction loss
            self.model.dof_damping[self._wheel_dof_indices] = (
                self._nominal_damping[self._wheel_dof_indices] * self.np_random.uniform(0.7, 1.5)
            )
            self.model.dof_frictionloss[self._wheel_dof_indices] = self.np_random.uniform(0.0005, 0.003, size=2)

            # 2. Actuator domain randomization
            self.motor_scales = self.np_random.uniform(0.85, 1.15, size=2)
            self.motor_deadband = float(self.np_random.uniform(0.0, 0.03))
            self.action_lag = float(self.np_random.uniform(0.0, 0.25))

            # 3. Sensor bias domain randomization (calibration offset & drift)
            self.pitch_bias = float(self.np_random.uniform(-0.035, 0.035))  # ~ +/- 2 deg
            self.gyro_bias = float(self.np_random.uniform(-0.02, 0.02))     # ~ +/- 1.1 deg/s
            self.pos_bias = float(self.np_random.uniform(-0.02, 0.02))      # ~ +/- 2 cm position offset
            self.wheel_vel_bias = self.np_random.uniform(-0.05, 0.05, size=2)
        else:
            self.motor_scales = np.ones(2)
            self.motor_deadband = 0.0
            self.action_lag = 0.0
            self.pitch_bias = 0.0
            self.gyro_bias = 0.0
            self.pos_bias = 0.0
            self.wheel_vel_bias = np.zeros(2)

        self.pitch_eq = self._balance_pitch()
        self.disturbance_steps = 0
        self.disturbance_force = 0.0
        self.prev_applied_action = np.zeros(2)

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Random initial lean about the wheel axle (y axis) and random heading.
        pitch = self.pitch_eq + self.np_random.uniform(-self.max_init_pitch, self.max_init_pitch)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        qy = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
        qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        qpos[3:7] = _quat_mul(qz, qy)
        # Rotate about the axle so the wheels stay on the floor.
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, qpos[3:7])
        qpos[0:3] = np.array([0.0, 0.0, self._wheel_radius]) - rot.reshape(3, 3) @ self._axle
        # Random initial position offset (-15 cm to +15 cm) in robot_pos mode
        # Teaches policy to recover and return to origin from anywhere nearby
        if self.obs_mode == "robot_pos":
            qpos[0] += float(self.np_random.uniform(-0.15, 0.15))

        qvel[3:6] = self.np_random.normal(0.0, 0.1, size=3)  # small angular kick

        self.set_state(qpos, qvel)

        self.cmd_vel = self.np_random.uniform(-0.5, 0.5) if self.command_vel else 0.0
        self.prev_action = np.zeros(2)
        return self._get_obs()


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
