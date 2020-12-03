import gym
import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 20000

# Create environment
env = gym.make('InvertedPendulum-v2')

# Instantiate the agent
optuna = True

if optuna:
  print("Optuna = True")
  file = "/saved_models/optuna_InvertedPendulum"
  model = A2C('MlpPolicy', env, 
        gamma = 0.9998711578038559,
        n_steps = 1024,
        vf_coef = 0.5,
        ent_coef = 0.03137755757220838,
        max_grad_norm = 0.33124753710629307,
        learning_rate = 0.010249423072591952,
        gae_lambda = 0.8572939586269473,       
        rms_prop_eps = 1e-5,
        verbose=1
        )
else:
  print("Optuna = False")
  file = "/saved_models/default_InvertedPendulum"
  model = A2C('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=TOTAL_TIMESTEPS)
model.save(os.getcwd() + file)

env.close()

