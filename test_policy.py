"""Test and visualize the trained balance bot policy in MuJoCo.

Usage:
    # 1. Run interactive 3D viewer with the best model:
    python test_policy.py

    # 2. Test a specific model file:
    python test_policy.py runs/ppo_pos_stay_origin/best_model.zip

    # 3. Run headless benchmark (no window, print detailed telemetry & statistics):
    python test_policy.py --no-render --steps 1000

    # 4. Test disturbance rejection (displace robot away from origin):
    python test_policy.py --push
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

import gymnasium as gym
from stable_baselines3 import PPO

import balance_bot  # noqa: F401 registers BalanceBot-v0


def find_default_model() -> Path:
    """Find the best trained model in runs/ directory."""
    candidates = [
        Path("runs/ppo_full/best_model.zip"),
        Path("runs/ppo_full/final_model.zip"),
        Path("runs/ppo_pos_stay_origin/best_model.zip"),
        Path("runs/ppo_pos_stay_origin/final_model.zip"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Fallback: scan runs/ for any best_model.zip
    runs = list(Path("runs").glob("*/best_model.zip"))
    if runs:
        runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return runs[0]
    raise FileNotFoundError("No trained model (.zip) found in runs/ directory!")


def main():
    parser = argparse.ArgumentParser(description="Test and visualize Balance Bot RL Policy")
    parser.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Path to trained SB3 model (.zip). Defaults to runs/ppo_pos_stay_origin/best_model.zip",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=1000, help="Max steps per episode (1000 steps ~ 33.3 sec)")
    parser.add_argument("--no-render", action="store_true", help="Disable 3D viewer (run headless for fast stats)")
    parser.add_argument("--push", action="store_true", help="Inject position disturbances to test origin centering")
    parser.add_argument("--obs-mode", default=None, choices=["robot_pos", "robot", "full"],
                        help="Override observation mode (default auto-detected from model)")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else find_default_model()
    print(f"\n=======================================================")
    print(f" Loading model: {model_path}")
    print(f"=======================================================")

    model = PPO.load(str(model_path), device="cpu")
    obs_dim = model.observation_space.shape[0]

    # Auto-detect observation configuration
    if args.obs_mode is not None:
        obs_mode = args.obs_mode
        command_vel = False
    elif obs_dim == 5:
        # 5-observation policy: [pitch, pitch_rate, left_vel, right_vel, x_pos]
        obs_mode = "robot_pos"
        command_vel = False
    elif obs_dim == 4:
        # 4-observation policy: [pitch, pitch_rate, left_vel, right_vel]
        obs_mode = "robot"
        command_vel = False
    elif obs_dim == 6:
        obs_mode = "robot_pos"
        command_vel = True
    else:
        obs_mode = "full"
        command_vel = False

    print(f"Policy observation dimension: {obs_dim} -> obs_mode='{obs_mode}', command_vel={command_vel}")
    render_mode = None if args.no_render else "human"

    env = gym.make(
        "BalanceBot-v0",
        render_mode=render_mode,
        obs_mode=obs_mode,
        command_vel=command_vel,
        push_disturbances=False,
        max_episode_steps=args.steps,
    )

    for ep in range(args.episodes):
        obs, _ = env.reset()
        
        # If test push requested, initialize robot with position offset
        if args.push:
            offset = 0.15 if ep % 2 == 0 else -0.15
            env.unwrapped.data.qpos[0] = offset
            import mujoco
            mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            obs = env.unwrapped._get_obs()
            print(f"\n[Episode {ep + 1}] Injected initial position offset: {offset:+.2f} m")
        else:
            print(f"\n[Episode {ep + 1}] Starting balance test from origin...")

        total_reward = 0.0
        step_count = 0
        x_positions = []
        pitches = []

        done = False
        print(f"{'Step':>6} | {'Pitch (deg)':>11} | {'Pitch Err':>10} | {'X Pos (cm)':>11} | {'Wheel Vel':>10} | {'Action [L, R]':>15}")
        print("-" * 78)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1

            x_pos = float(env.unwrapped.data.qpos[0])
            pitch_deg = float(np.degrees(env.unwrapped._pitch_roll()[0]))
            pitch_eq_deg = float(np.degrees(env.unwrapped.pitch_eq))
            pitch_err_deg = pitch_deg - pitch_eq_deg
            wheel_vel = float(env.unwrapped._forward_vel())

            x_positions.append(x_pos)
            pitches.append(pitch_err_deg)

            # Print telemetry every 50 steps (~1.6s) or on final step
            if step_count % 50 == 0 or step_count == 1:
                print(
                    f"{step_count:>6} | "
                    f"{pitch_deg:>9.2f} deg | "
                    f"{pitch_err_deg:>8.2f} deg | "
                    f"{x_pos * 100.0:>10.2f} cm | "
                    f"{wheel_vel:>9.2f} m/s | "
                    f"[{action[0]:>6.2f}, {action[1]:>6.2f}]"
                )

            done = terminated or truncated

        # Summary for episode
        x_arr = np.array(x_positions)
        mean_abs_x = float(np.mean(np.abs(x_arr))) * 100.0
        max_abs_x = float(np.max(np.abs(x_arr))) * 100.0
        mean_abs_pitch = float(np.mean(np.abs(pitches)))

        status = "FELL OVER" if terminated and step_count < args.steps else "SUCCESS (Balanced entire run)"
        print("-" * 78)
        print(f"Result: {status}")
        print(f"  - Survived Steps:    {step_count} / {args.steps} ({step_count * env.unwrapped.dt:.1f} seconds)")
        print(f"  - Mean |X| Drift:    {mean_abs_x:.2f} cm")
        print(f"  - Max |X| Drift:     {max_abs_x:.2f} cm")
        print(f"  - Final X Position:  {x_positions[-1] * 100.0:+.2f} cm")
        print(f"  - Mean Pitch Error:  {mean_abs_pitch:.2f} deg")
        print(f"  - Total Return:      {total_reward:.1f}")

    env.close()
    print("\nTest completed successfully.\n")


if __name__ == "__main__":
    main()
