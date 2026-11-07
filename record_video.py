"""Record video / GIF snapshot of the robot balancing in MuJoCo.

Useful for GitHub READMEs, project showcases, and documentation.

Usage:
    # 1. Record 6-second balancing GIF from newest trained model:
    python record_video.py

    # 2. Record specific model to MP4 and GIF with a push disturbance:
    python record_video.py runs/ppo_full/best_model.zip --duration 6 --push

    # 3. Custom output path and size:
    python record_video.py --output docs/assets/robot_balance.gif --width 640 --height 480
"""

import argparse
from pathlib import Path
import time
import imageio
import mujoco
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

import balance_bot  # noqa: F401
from balance_bot.env import MODEL_PATH


def find_default_model() -> Path:
    """Find the best trained model in runs/ directory."""
    runs = sorted(Path("runs").glob("*/best_model.zip"), key=lambda p: p.stat().st_mtime)
    if runs:
        return runs[-1]
    fallback = Path("runs/ppo_full/best_model.zip")
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("No trained model found in runs/ directory!")


def get_obs_for_model(env, obs_dim: int) -> np.ndarray:
    """Extract matching observation vector from environment unwrapped state."""
    u = env.unwrapped
    # Query sensor data
    pitch = float(u.data.sensor("chassis_pitch").data[0]) + u.pitch_bias
    gyro_y = float(u.data.sensor("gyro_y").data[0]) + u.gyro_bias
    left_vel = float(u.data.sensor("left_wheel_vel").data[0]) * 0.1
    right_vel = float(u.data.sensor("right_wheel_vel").data[0]) * 0.1
    pos_x = float(u.data.qpos[0])

    if obs_dim == 4:
        return np.array([pitch, gyro_y, left_vel, right_vel], dtype=np.float32)
    elif obs_dim == 5:
        # Check if 5-dim model trained with cmd_vel or pos_error
        # Default robot_pos has pos_error as 5th element
        return np.array([pitch, gyro_y, left_vel, right_vel, pos_x], dtype=np.float32)
    elif obs_dim == 6:
        cmd_vel = getattr(u, "cmd_vel", 0.0)
        return np.array([pitch, gyro_y, left_vel, right_vel, pos_x, cmd_vel], dtype=np.float32)
    elif obs_dim == 10:
        roll = float(u.data.sensor("chassis_roll").data[0])
        roll_rate = float(u.data.sensor("gyro_x").data[0])
        yaw_rate = float(u.data.sensor("gyro_z").data[0])
        cmd_vel = getattr(u, "cmd_vel", 0.0)
        prev_a = getattr(u, "prev_action", np.zeros(2))
        return np.array([
            pitch, roll, gyro_y, roll_rate, yaw_rate,
            left_vel * 10, right_vel * 10,
            cmd_vel, prev_a[0], prev_a[1]
        ], dtype=np.float32)
    return np.zeros(obs_dim, dtype=np.float32)


def record(
    model_path: Path,
    output_path: Path,
    duration: float = 6.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    push: bool = True,
):
    print(f"Loading policy: {model_path}")
    policy = PPO.load(model_path, device="cpu")
    obs_dim = policy.observation_space.shape[0]
    print(f"Policy observation dimension: {obs_dim}")

    import gymnasium as gym
    if obs_dim == 6:
        obs_mode, command_vel = "robot_pos", True
    elif obs_dim == 5:
        obs_mode, command_vel = "robot_pos", False
    elif obs_dim == 10:
        obs_mode, command_vel = "full", False
    else:
        obs_mode, command_vel = "robot", False

    env = gym.make("BalanceBot-v0", obs_mode=obs_mode, command_vel=command_vel, push_disturbances=False)
    u = env.unwrapped
    m = u.model
    d = u.data

    renderer = mujoco.Renderer(m, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.distance = 1.15
    camera.elevation = -14.0
    camera.azimuth = 65.0
    camera.lookat = [0.0, 0.0, 0.08]

    obs, _ = env.reset()
    frames = []
    total_steps = int(duration / u.dt)
    frame_interval = max(1, int(1.0 / (fps * u.dt)))

    push_start = int(total_steps * 0.4)
    push_end = push_start + int(0.12 / u.dt)

    print(f"Simulating & rendering {duration}s ({total_steps} steps at {int(1/u.dt)} Hz)...")

    for step in range(total_steps):
        # Apply external disturbance push if requested
        if push and push_start <= step < push_end:
            u.data.xfrc_applied[u._chassis_id, 0] = 3.2  # 3.2 N impulse shove
        else:
            u.data.xfrc_applied[u._chassis_id, :] = 0.0

        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        if step % frame_interval == 0:
            # Smoothly orbit camera slightly for cinematic effect
            camera.azimuth = 65.0 + 15.0 * np.sin(step / total_steps * np.pi)
            camera.lookat[0] = float(d.qpos[0]) * 0.6  # track robot slightly
            renderer.update_scene(d, camera=camera)
            img = renderer.render()
            frames.append(img)

        if terminated:
            print("Robot fell during recording! Resetting...")
            obs, _ = env.reset()

    env.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as GIF
    gif_path = output_path.with_suffix(".gif")
    print(f"Generating optimized GIF: {gif_path} ({len(frames)} frames at {fps} fps)...")
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    print(f"[Done] Saved GIF to: {gif_path} ({gif_path.stat().st_size / 1024:.1f} KB)")

    # Save as MP4 if imageio-ffmpeg is available
    try:
        mp4_path = output_path.with_suffix(".mp4")
        writer = imageio.get_writer(str(mp4_path), fps=fps, quality=8)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[Done] Saved MP4 to: {mp4_path} ({mp4_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[Notice] MP4 export skipped: {e}")

    return gif_path


def main():
    parser = argparse.ArgumentParser(description="Record video/GIF of balance robot for GitHub")
    parser.add_argument("model", nargs="?", default=None, help="Path to policy .zip")
    parser.add_argument("-o", "--output", default="docs/assets/balance_bot.gif", help="Output file path (.gif or .mp4)")
    parser.add_argument("--duration", type=float, default=6.0, help="Duration in seconds (default: 6.0)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--no-push", action="store_true", help="Disable disturbance push")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_default_model()
    record(
        model_path=model_path,
        output_path=Path(args.output),
        duration=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        push=not args.no_push,
    )


if __name__ == "__main__":
    main()
