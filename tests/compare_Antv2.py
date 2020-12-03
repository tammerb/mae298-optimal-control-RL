import gym
import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 300000

# Create environment
env = gym.make('Ant-v2')

# Instantiate the agent
optuna = True

if optuna:
  print("Optuna = True")
  file = "/saved_models/optuna_Ant"
  model = A2C('MlpPolicy', env, 
        gamma = 0.9849155664416022,
        n_steps = 32,
        vf_coef = 0.5,
        ent_coef = 8.54404882554226e-07,
        max_grad_norm = 3.177649963510794,
        learning_rate = 3.2812686482837234e-05,
        gae_lambda = 0.995288127336676,       
        rms_prop_eps = 1e-5,
        verbose=1
        )
else:
  print("Optuna = False")
  file = "/saved_models/default_Ant"
  model = A2C('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(os.getcwd() + file)

env.close()

