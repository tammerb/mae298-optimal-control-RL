########################################
# This script creates a CustomAnt env object,
# and a baselines model object (PPO, A2C, etc)
# Evaluates (rolls out) the untrained model (random actions),
# then train the model and reevaulates it (rollout again).
# It saves both model rollouts with model.save.
########################################

import gym
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from PlotCallBack import PlotCallBack
from stable_baselines3.common.monitor import Monitor
import custom_ant

log_dir = "tmp_blockv3/"
os.makedirs(log_dir, exist_ok=True)

timesteps = 10000000
env = gym.make('Block-v3')
env = Monitor(env, log_dir)
model = A2C('MlpPolicy', env, verbose=1, gamma=0.97686, n_steps=64)

mean_reward_before = 0#, _ = evaluate_policy(model, env,n_eval_episodes=1000)
model.save(os.getcwd() + "/block_three_leg_lift_model_untrained")

callback = PlotCallBack(check_freq=1000, log_dir=log_dir)
model.learn(total_timesteps=int(timesteps), callback=callback)
model.save(os.getcwd() + "/block_three_leg_lift_model_10M")
mean_reward_after, _ = evaluate_policy(model, env,n_eval_episodes=10)

print("Mean reward before learning:", mean_reward_before)
print("Mean reward after learning:", mean_reward_after)
plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Block-v3")
plt.show()