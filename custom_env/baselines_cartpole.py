import gym
import custom_ant
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO2
from stable_baselines import A2C


env = gym.make('CartPole-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
#model = A2C(MlpPolicy, env, verbose=1, gamma=0.9, n_steps=20)

env = model.get_env()
mean_reward_0, std_reward_0 = evaluate_policy(model, env, n_eval_episodes=500)

model.learn(total_timesteps=1000)

mean_reward_1, std_reward_1 = evaluate_policy(model, env, n_eval_episodes=500)

print(f"before learning mean_reward:{mean_reward_0:.2f} +/- {std_reward_0:.2f}")

print(f"after learning mean_reward:{mean_reward_1:.2f} +/- {std_reward_1:.2f}")
