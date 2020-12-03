import gym
import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 20000

# Create environment
env = gym.make('CartPole-v1')

# Instantiate the agent
optuna = False

if optuna:
  print("Optuna = True")
  file = "/saved_models/optuna_cartpole"
  model = A2C('MlpPolicy', env, 
        gamma = 0.9987940879356862,
        n_steps = 256,
        vf_coef = 0.5,
        ent_coef = 1.2954655366837278e-06,
        max_grad_norm = 0.7111167513074961,
        learning_rate = 0.0015613269950806646,
        gae_lambda = 0.994637205730802,       
        rms_prop_eps = 1e-5,
        verbose=1
        )
else:
  print("Optuna = False")
  file = "/saved_models/default_cartpole"
  model = A2C('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(os.getcwd() + file)

env.close()