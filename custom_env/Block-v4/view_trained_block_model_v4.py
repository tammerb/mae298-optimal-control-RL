########################################
# This script loads the models saved in train_ant_model
# and prints the reward functions and renders the two rollouts back to back.
########################################

import os
import gym
import custom_ant

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

best_model = A2C.load(os.getcwd() + "/bag/optuna/best_model", verbose=1)

env = gym.make('Block-v4')

best_model.set_env(DummyVecEnv([lambda: env]))

cumulative_reward = 0
obs = env.reset()
for i in range(1000):
    action, _states = best_model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cumulative_reward += rewards
    env.render()

#print("Untrained total reward:", cumulative_reward_untrained)
print("Trained Total reward:", cumulative_reward)
