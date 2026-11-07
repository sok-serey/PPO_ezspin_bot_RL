"""Train a PPO balancing policy.

    python train.py --timesteps 2000000 --n-envs 8
    python train.py --command-vel          # also learn to follow a forward speed command
"""

import argparse
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import balance_bot  # noqa: F401  (registers BalanceBot-v0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--command-vel", action="store_true", help="randomize a forward velocity command")
    p.add_argument(
        "--obs-mode",
        choices=["robot", "robot_pos", "full"],
        default="robot_pos",
        help="observation mode: 'robot_pos' (5 obs with stay-in-place position hold), 'robot' (4 obs) or 'full' (10 obs)",
    )
    p.add_argument("--no-domain-rand", action="store_true", help="disable domain randomization")
    p.add_argument("--alive-bonus", type=float, default=1.0, help="reward bonus for staying upright per step")
    p.add_argument("--pitch-penalty-coef", type=float, default=5.0, help="penalty multiplier for pitch error^2")
    p.add_argument("--pitch-rate-penalty-coef", type=float, default=0.05, help="penalty multiplier for gyro pitch rate^2")
    p.add_argument("--action-penalty-coef", type=float, default=0.01, help="penalty multiplier for motor action^2")
    p.add_argument("--position-penalty-coef", type=float, default=1.5, help="penalty multiplier for (x^2 + y^2) drift")
    p.add_argument("--yaw-penalty-coef", type=float, default=0.1, help="penalty multiplier for yaw rotation rate")
    p.add_argument("--max-pos-distance", type=float, default=0.8, help="max allowed distance from origin before termination")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    run_dir = args.run_dir or Path("runs") / datetime.now().strftime("ppo_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = {
        "command_vel": args.command_vel,
        "obs_mode": args.obs_mode,
        "randomize_dynamics": not args.no_domain_rand,
        "alive_bonus": args.alive_bonus,
        "pitch_penalty_coef": args.pitch_penalty_coef,
        "pitch_rate_penalty_coef": args.pitch_rate_penalty_coef,
        "action_penalty_coef": args.action_penalty_coef,
        "position_penalty_coef": args.position_penalty_coef,
        "yaw_penalty_coef": args.yaw_penalty_coef,
        "max_pos_distance": args.max_pos_distance,
    }

    env = make_vec_env("BalanceBot-v0", n_envs=args.n_envs, seed=args.seed, env_kwargs=env_kwargs,
                       vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None)
    eval_kwargs = dict(env_kwargs)
    eval_kwargs["push_disturbances"] = False  # clean evaluation metrics without external kicks
    eval_env = make_vec_env("BalanceBot-v0", n_envs=1, seed=args.seed + 1000, env_kwargs=eval_kwargs)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[64, 64]),
        tensorboard_log=str(run_dir / "tb"),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    callbacks = [
        EvalCallback(eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir),
                     eval_freq=max(20_000 // args.n_envs, 1), n_eval_episodes=5, deterministic=True),
        CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1), save_path=str(run_dir / "checkpoints")),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save(run_dir / "final_model")
    print(f"Saved to {run_dir}")

    # Automatically export ONNX for robot deployment
    try:
        from export_onnx import export_to_onnx
        if (run_dir / "best_model.zip").exists():
            export_to_onnx(run_dir / "best_model.zip")
        export_to_onnx(run_dir / "final_model.zip")
    except Exception as e:
        print(f"ONNX auto-export skipped or failed: {e}")

    # Automatically generate demo video/GIF snapshot for GitHub
    try:
        from record_video import record
        best_path = run_dir / "best_model.zip"
        if best_path.exists():
            record(best_path, run_dir / "snapshot.gif", duration=4.0, fps=25)
    except Exception as e:
        print(f"Snapshot GIF generation skipped: {e}")


if __name__ == "__main__":
    main()
