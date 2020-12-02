#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 20:28:43 2020

@author: thomas

The function in this script open a model and trains it for the given number of timesteps
"""


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
import custom_ant
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C
import os

def Continue_Training(env_name='CustomAnt-v0',model_name='ant_model',timesteps=100000):
    print(os.getcwd() + "/" + model_name)
    env = gym.make(env_name)  #Create enviroment
    model = A2C(MlpPolicy, env, verbose=1, gamma=0.9, n_steps=20)
    model.load_parameters(os.getcwd() + "/" + model_name)
    model.learn(total_timesteps=timesteps,reset_num_timesteps=False)
    model.save(os.getcwd() + "/" + model_name)
    return model