########################################
# This script loads the models saved in train_ant_model
# and prints the reward functions and renders the two rollouts back to back.
########################################

import os
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

from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

model_untrained = A2C.load(os.getcwd() + "/block_model_untrained", verbose=1)
model = A2C.load(os.getcwd() + "/block_model_2M", verbose=1)


env = gym.make('Block-v0')

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
