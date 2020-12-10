########################################
# This script creates a CustomAnt env object,
# and a baselines model object (PPO, A2C, etc)
# Evaluates (rolls out) the untrained model (random actions),
# then train the model and reevaulates it (rollout again).
# It saves both model rollouts with model.save.
########################################

import gym
import os
import custom_ant
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
from PlotCallBack import PlotCallBack
from stable_baselines3.common.monitor import Monitor


timesteps = 40000

log_dir = "tmp2/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make('Block-v2')
env = Monitor(env, log_dir)

model = A2C('MlpPolicy', env, verbose=1, gamma=0.97686, n_steps=64)

callback = PlotCallBack(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=int(timesteps), callback=callback)


model.save(os.getcwd() + "/block_go_to_location_trained_sb3")
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Block-v2")
plt.show()

