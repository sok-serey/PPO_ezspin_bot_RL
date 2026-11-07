from gymnasium.envs.registration import register

from balance_bot.env import BalanceBotEnv

register(
    id="BalanceBot-v0",
    entry_point="balance_bot.env:BalanceBotEnv",
    max_episode_steps=1000,  # 10 s at 100 Hz
)

__all__ = ["BalanceBotEnv"]
