########################################
# This script loads the models saved in train_ant_model
# and prints the reward functions and renders the two rollouts back to back.
########################################

import os
import gym
import custom_ant

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

model_untrained = PPO2.load(os.getcwd() + "/ant_model_untrained", verbose=1)
model = PPO2.load(os.getcwd() + "/ant_model", verbose=1)


env = gym.make('CustomAnt-v0')

model_untrained.set_env(DummyVecEnv([lambda: env]))
model.set_env(DummyVecEnv([lambda: env]))


cumulative_reward_untrained = 0
obs = env.reset()
for i in range(1000):
    action, _states = model_untrained.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cumulative_reward_untrained += rewards
    env.render()

cumulative_reward = 0
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cumulative_reward += rewards
    env.render()

print("Untrained total reward:", cumulative_reward_untrained)
print("Trained Total reward:", cumulative_reward)
