"""Run a trained policy in the MuJoCo viewer.

    python play.py runs/<run>/best_model.zip
    python play.py runs/<run>/best_model.zip --cmd-vel 0.3
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO

import balance_bot  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?", default=None, help="path to a saved SB3 .zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--cmd-vel", type=float, default=None, help="fixed forward speed command (m/s)")
    p.add_argument("--n-robots", type=int, default=1, help="number of parallel robots to view (e.g. 8)")
    args = p.parse_args()

    if args.n_robots > 1 or args.model is None:
        import sys
        import play_parallel
        sys.argv = ["play_parallel.py"]
        if args.model:
            sys.argv.append(args.model)
        sys.argv.extend(["--n-robots", str(args.n_robots)])
        if args.cmd_vel is not None:
            sys.argv.extend(["--cmd-vel", str(args.cmd_vel)])
        play_parallel.main()
        return

    model = PPO.load(args.model, device="cpu")
    obs_dim = model.observation_space.shape[0]
    if obs_dim == 10:
        obs_mode = "full"
        command_vel = False
    elif obs_dim == 5:
        if args.cmd_vel is not None:
            obs_mode = "robot"
            command_vel = True
        else:
            obs_mode = "robot_pos"
            command_vel = False
    elif obs_dim == 6:
        obs_mode = "robot_pos"
        command_vel = True
    else:
        obs_mode = "robot"
        command_vel = False

    env = gym.make(
        "BalanceBot-v0",
        render_mode="human",
        command_vel=command_vel,
        obs_mode=obs_mode,
        push_disturbances=False,
    )

    for ep in range(args.episodes):
        obs, _ = env.reset()
        if args.cmd_vel is not None:
            env.unwrapped.cmd_vel = args.cmd_vel
        total, steps, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        print(f"episode {ep}: return={total:.1f} steps={steps} fell={terminated}")
    env.close()


if __name__ == "__main__":
    main()
