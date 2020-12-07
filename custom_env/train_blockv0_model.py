########################################
# This script creates a CustomAnt env object,
# and a baselines model object (PPO, A2C, etc)
# Evaluates (rolls out) the untrained model (random actions),
# then train the model and reevaulates it (rollout again).
# It saves both model rollouts with model.save.
########################################

import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'CustomAnt-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    elif 'CustomAnt-v2' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    elif 'CustomAnt-v3' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
    elif 'Block-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
import custom_ant
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C

import os

env = gym.make('Block-v0')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

#model = PPO2(MlpPolicy, env, verbose=1)
model = A2C(MlpPolicy, env, verbose=1, gamma=0.97686, n_steps=64,lr_schedule='linear')

cumulative_reward_before = 0

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cumulative_reward_before += rewards

model.save(os.getcwd() + "/block_model_untrained")

model.learn(total_timesteps=2000000,reset_num_timesteps=False)

cumulative_reward_after = 0

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    cumulative_reward_after += rewards

model.save(os.getcwd() + "/block_model_2M")

print("Total reward before learning:", cumulative_reward_before)
print("Total reward after learning:", cumulative_reward_after)