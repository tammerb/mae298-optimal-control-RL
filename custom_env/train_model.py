import gym
import custom_ant
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C

import os



env = gym.make('CustomAnt-v2')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
#model = A2C(MlpPolicy, env, verbose=1, gamma=0.9, n_steps=20)

cumulative_reward_before = 0

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cumulative_reward_before += rewards

model.save(os.getcwd() + "/ant_model_untrained")

model.learn(total_timesteps=100000)

cumulative_reward_after = 0

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cumulative_reward_after += rewards

model.save(os.getcwd() + "/ant_model")

print("Total reward before learning:", cumulative_reward_before)
print("Total reward after learning:", cumulative_reward_after)