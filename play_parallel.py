"""Multi-robot parallel viewer for MuJoCo.

Simulates and displays N robots (default 8) in parallel in a single MuJoCo 3D window.
Each robot runs independent physics, domain randomization, and policy inference.

Usage:
    # 1. Watch latest training run in real-time (auto-reloads new checkpoints):
    python play_parallel.py

    # 2. View a specific model with 8 robots:
    python play_parallel.py runs/ppo_20261028_164718/best_model.zip --n-robots 8

    # 3. View with velocity commands:
    python play_parallel.py runs/my_run/best_model.zip --cmd-vel 0.3
"""

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).parent
BASE_XML = ROOT / "balance_bot" / "assets" / "balance_bot.xml"

# Distinct vibrant colors for up to 8+ robots
ROBOT_COLORS = [
    [0.2, 0.6, 1.0, 1.0],   # Blue
    [1.0, 0.35, 0.35, 1.0], # Coral Red
    [0.3, 0.85, 0.45, 1.0], # Emerald Green
    [1.0, 0.7, 0.15, 1.0],  # Orange
    [0.75, 0.4, 0.95, 1.0], # Purple
    [0.15, 0.85, 0.85, 1.0],# Cyan
    [0.95, 0.4, 0.75, 1.0], # Pink
    [0.9, 0.9, 0.25, 1.0],  # Yellow
]


def build_multi_robot_xml(n_robots: int = 8, spacing_y: float = 0.7) -> str:
    """Dynamically generates a MuJoCo XML string with n_robots arranged along the Y axis."""
    tree = ET.parse(BASE_XML)
    root = tree.getroot()

    worldbody = root.find("worldbody")
    chassis_orig = worldbody.find("./body[@name='chassis']")
    contact = root.find("contact")
    actuator = root.find("actuator")
    sensor = root.find("sensor")

    worldbody.remove(chassis_orig)
    contact.clear()
    actuator.clear()
    sensor.clear()

    # Point mesh directory to absolute path
    compiler = root.find("compiler")
    compiler.set("meshdir", str((BASE_XML.parent / "meshes").resolve()))

    for i in range(n_robots):
        prefix = f"bot{i}_"
        y_offset = (i - (n_robots - 1) / 2.0) * spacing_y
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]

        new_chassis = ET.fromstring(ET.tostring(chassis_orig))
        new_chassis.set("name", f"{prefix}chassis")
        new_chassis.set("pos", f"0 {y_offset:.4f} 0.0604")
        new_chassis.find("./freejoint").set("name", f"{prefix}root")

        # Color the base visual mesh
        for g in new_chassis.iter("geom"):
            if g.get("class") == "visual" and g.get("mesh") == "base_link":
                g.set("rgba", f"{color[0]} {color[1]} {color[2]} {color[3]}")
            if g.get("name"):
                g.set("name", f"{prefix}{g.get('name')}")

        for b in new_chassis.iter("body"):
            if b.get("name") == "left_wheel_link":
                b.set("name", f"{prefix}left_wheel_link")
                b.find("./joint[@name='left_wheel_joint']").set("name", f"{prefix}left_wheel_joint")
            elif b.get("name") == "right_wheel_link":
                b.set("name", f"{prefix}right_wheel_link")
                b.find("./joint[@name='right_wheel_joint']").set("name", f"{prefix}right_wheel_joint")

        for s in new_chassis.iter("site"):
            if s.get("name"):
                s.set("name", f"{prefix}{s.get('name')}")

        worldbody.append(new_chassis)

        # Contact exclusions between chassis and wheels
        ET.SubElement(contact, "exclude", body1=f"{prefix}chassis", body2=f"{prefix}left_wheel_link")
        ET.SubElement(contact, "exclude", body1=f"{prefix}chassis", body2=f"{prefix}right_wheel_link")

        # Actuators
        ET.SubElement(actuator, "motor", name=f"{prefix}left", joint=f"{prefix}left_wheel_joint", gear="0.3", ctrlrange="-1 1")
        ET.SubElement(actuator, "motor", name=f"{prefix}right", joint=f"{prefix}right_wheel_joint", gear="0.3", ctrlrange="-1 1")

        # Sensors
        ET.SubElement(sensor, "framequat", name=f"{prefix}imu_quat", objtype="site", objname=f"{prefix}imu")
        ET.SubElement(sensor, "gyro", name=f"{prefix}imu_gyro", site=f"{prefix}imu")
        ET.SubElement(sensor, "jointvel", name=f"{prefix}wheel_left_vel", joint=f"{prefix}left_wheel_joint")
        ET.SubElement(sensor, "jointvel", name=f"{prefix}wheel_right_vel", joint=f"{prefix}right_wheel_joint")

    return ET.tostring(root, encoding="unicode")


def reset_robot(i: int, model: mujoco.MjModel, data: mujoco.MjData, state: dict, y_offset: float,
                randomize: bool = True, max_init_pitch: float = 0.15):
    """Resets the state of a single robot independently."""
    p = f"bot{i}_"
    if randomize:
        state["pitch_bias"] = float(np.random.uniform(-0.035, 0.035))
        state["gyro_bias"] = float(np.random.uniform(-0.02, 0.02))
        state["wheel_vel_bias"] = np.random.uniform(-0.05, 0.05, size=2)
        state["motor_scales"] = np.random.uniform(0.85, 1.15, size=2)
        state["motor_deadband"] = float(np.random.uniform(0.0, 0.025))
        state["action_lag"] = float(np.random.uniform(0.0, 0.2))
    else:
        state["pitch_bias"] = 0.0
        state["gyro_bias"] = 0.0
        state["wheel_vel_bias"] = np.zeros(2)
        state["motor_scales"] = np.ones(2)
        state["motor_deadband"] = 0.0
        state["action_lag"] = 0.0

    state["prev_action"] = np.zeros(2)
    state["prev_applied"] = np.zeros(2)
    state["ep_len"] = 0
    state["respawn_timer"] = 0
    state["fell"] = False

    wheel_radius = float(model.geom(f"{p}wheel_left").size[0])
    axle = model.body(f"{p}left_wheel_link").pos.copy()
    axle[1] = 0.0

    # Calculate theoretical balance pitch for this robot
    com, mass = np.zeros(3), 0.0
    for bname in ("chassis", "left_wheel_link", "right_wheel_link"):
        b = model.body(f"{p}{bname}")
        offset = b.pos if bname != "chassis" else 0.0
        com += b.mass[0] * (offset + b.ipos)
        mass += b.mass[0]
    dx, _, dz = com / mass - axle
    pitch_eq = float(np.arctan2(-dx, dz))
    state["pitch_eq"] = pitch_eq

    pitch = pitch_eq + np.random.uniform(-max_init_pitch, max_init_pitch)
    yaw = np.random.uniform(-np.pi, np.pi)
    qy = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    quat = np.array([
        qz[0]*qy[0] - qz[1]*qy[1] - qz[2]*qy[2] - qz[3]*qy[3],
        qz[0]*qy[1] + qz[1]*qy[0] + qz[2]*qy[3] - qz[3]*qy[2],
        qz[0]*qy[2] - qz[1]*qy[3] + qz[2]*qy[0] + qz[3]*qy[1],
        qz[0]*qy[3] + qz[1]*qy[2] - qz[2]*qy[1] + qz[3]*qy[0],
    ])
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    pos = np.array([0.0, y_offset, wheel_radius]) - rot.reshape(3, 3) @ axle

    root_j = model.joint(f"{p}root").id
    qpos_adr = model.jnt_qposadr[root_j]
    qvel_adr = model.jnt_dofadr[root_j]

    data.qpos[qpos_adr : qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
    data.qvel[qvel_adr : qvel_adr + 6] = np.random.normal(0.0, 0.05, size=6)

    for w in ("left_wheel_joint", "right_wheel_joint"):
        jid = model.joint(f"{p}{w}").id
        data.qpos[model.jnt_qposadr[jid]] = 0.0
        data.qvel[model.jnt_dofadr[jid]] = 0.0


def get_robot_obs(i: int, data: mujoco.MjData, state: dict, obs_dim: int, cmd_vel: float) -> np.ndarray:
    """Constructs the observation vector for robot i matching the policy's dimension."""
    p = f"bot{i}_"
    w, x, y, z = data.sensor(f"{p}imu_quat").data
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    gyro = data.sensor(f"{p}imu_gyro").data
    wl = data.sensor(f"{p}wheel_left_vel").data[0]
    wr = data.sensor(f"{p}wheel_right_vel").data[0]

    if obs_dim in (4, 5):
        # 4-dim robot observation: pitch, pitch_rate, left_wheel_vel, right_wheel_vel
        obs = [
            pitch + state["pitch_bias"],
            gyro[1] + state["gyro_bias"],
            (wl + state["wheel_vel_bias"][0]) * 0.1,
            (wr + state["wheel_vel_bias"][1]) * 0.1,
        ]
        if obs_dim == 5:
            obs.append(cmd_vel)
        return np.array(obs, dtype=np.float64)
    else:
        # 10-dim legacy observation
        return np.array([
            pitch + state["pitch_bias"],
            roll,
            gyro[1] + state["gyro_bias"],
            gyro[0],
            gyro[2],
            (wl + state["wheel_vel_bias"][0]) * 0.1,
            (wr + state["wheel_vel_bias"][1]) * 0.1,
            cmd_vel,
            *state["prev_action"],
        ], dtype=np.float64)


def apply_robot_action(i: int, data: mujoco.MjData, action: np.ndarray, state: dict):
    """Applies control torque to robot i with motor scaling and lag."""
    p = f"bot{i}_"
    action = np.clip(action, -1.0, 1.0)
    scaled = action * state["motor_scales"]
    if state["motor_deadband"] > 0:
        mask = np.abs(scaled) < state["motor_deadband"]
        scaled[mask] = 0.0
    applied = (1.0 - state["action_lag"]) * scaled + state["action_lag"] * state["prev_applied"]
    state["prev_applied"] = applied.copy()
    state["prev_action"] = action.copy()
    data.actuator(f"{p}left").ctrl = applied[0]
    data.actuator(f"{p}right").ctrl = applied[1]


def find_latest_policy(model_arg: str | None) -> Path | None:
    if model_arg:
        p = Path(model_arg)
        return p if p.exists() else None
    found = sorted(ROOT.glob("runs/*/best_model.zip"), key=lambda p: p.stat().st_mtime)
    if not found:
        found = sorted(ROOT.glob("runs/*/final_model.zip"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def main():
    p = argparse.ArgumentParser(description="View 8 robots training or running in parallel in MuJoCo")
    p.add_argument("model", nargs="?", default=None, help="path to policy .zip (default: newest in runs/)")
    p.add_argument("--n-robots", type=int, default=8, help="number of parallel robots (default: 8)")
    p.add_argument("--spacing", type=float, default=0.7, help="spacing between robots in meters")
    p.add_argument("--cmd-vel", type=float, default=0.0, help="forward velocity command")
    p.add_argument("--no-domain-rand", action="store_true", help="disable domain randomization during viewing")
    p.add_argument("--no-watch", action="store_true", help="disable automatic policy reloading")
    args = p.parse_args()

    n_robots = args.n_robots
    spacing_y = args.spacing
    print(f"Building MuJoCo world with {n_robots} parallel robots...")
    xml_str = build_multi_robot_xml(n_robots=n_robots, spacing_y=spacing_y)
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    bot_states = [{} for _ in range(n_robots)]
    offsets_y = [(i - (n_robots - 1) / 2.0) * spacing_y for i in range(n_robots)]

    # Initial reset
    for i in range(n_robots):
        reset_robot(i, model, data, bot_states[i], offsets_y[i], randomize=not args.no_domain_rand)
    mujoco.mj_forward(model, data)

    current_policy = None
    current_policy_path = None
    current_policy_mtime = 0.0

    def maybe_reload():
        nonlocal current_policy, current_policy_path, current_policy_mtime
        path = find_latest_policy(args.model)
        if path is None or not path.exists():
            return
        mtime = path.stat().st_mtime
        if path == current_policy_path and mtime == current_policy_mtime:
            return
        try:
            current_policy = PPO.load(path, device="cpu")
            current_policy_path = path
            current_policy_mtime = mtime
            obs_dim = current_policy.observation_space.shape[0]
            print(f"[Loaded] {path.relative_to(ROOT)} (obs_dim={obs_dim})")
        except Exception as e:
            print(f"[Warning] Failed to load {path}: {e}")

    maybe_reload()

    print("\n" + "=" * 60)
    print(f"  MuJoCo Parallel 3D Viewer ({n_robots} Robots)")
    print("=" * 60)
    print("  Controls:")
    print("    - Spacebar:       Pause / Resume simulation")
    print("    - Right-click + Drag: Orbit camera")
    print("    - Scroll:         Zoom")
    print("    - Ctrl + Right-drag: Apply force / push a robot!")
    print("=" * 60 + "\n")

    frame_skip = 5  # 100 Hz control loop
    dt = model.opt.timestep * frame_skip  # 0.01s
    last_reload_check = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = max(3.5, n_robots * 0.65)
        viewer.cam.elevation = -22.0
        viewer.cam.azimuth = 90.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.15]

        next_t = time.perf_counter()
        while viewer.is_running():
            now = time.perf_counter()

            # Auto-reload policy if training is active in another process
            if not args.no_watch and (now - last_reload_check > 2.0):
                maybe_reload()
                last_reload_check = now

            obs_dim = current_policy.observation_space.shape[0] if current_policy else 4

            # Compute actions for each of the n_robots
            for i in range(n_robots):
                state = bot_states[i]
                if state.get("fell", False):
                    state["respawn_timer"] += 1
                    # After ~0.6 s fallen, stand back up
                    if state["respawn_timer"] >= 60:
                        reset_robot(i, model, data, state, offsets_y[i], randomize=not args.no_domain_rand)
                    continue

                obs = get_robot_obs(i, data, state, obs_dim, args.cmd_vel)
                if current_policy is not None:
                    action, _ = current_policy.predict(obs, deterministic=True)
                else:
                    action = np.zeros(2)

                apply_robot_action(i, data, action, state)
                state["ep_len"] += 1

                # Check fall condition
                p = f"bot{i}_"
                w, x, y, z = data.sensor(f"{p}imu_quat").data
                pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
                roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
                pitch_err = pitch - state["pitch_eq"]
                if abs(pitch_err) > 0.6 or abs(roll) > 0.6:
                    state["fell"] = True
                    state["respawn_timer"] = 0

            # Step physics by frame_skip sub-steps
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

            viewer.sync()

            next_t += dt
            time.sleep(max(0.0, next_t - time.perf_counter()))


if __name__ == "__main__":
    main()

