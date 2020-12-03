import gym
import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 1e6

# Create environment
env = gym.make('Ant-v2')

def train_model(optuna):
  if optuna:
    print("Optuna = True")
    file = "/saved_models/optuna_Ant"
    model = A2C('MlpPolicy', env, 
          gamma = 0.9980152091572712,
          n_steps = 256,
          vf_coef = 0.5,
          ent_coef = 1.7157437409384092e-06,
          max_grad_norm = 0.3598150713374998,
          learning_rate = 7.60205316312803e-05,
          gae_lambda = 0.942374289947569,       
          rms_prop_eps = 1e-5,
          verbose=0
          )
  else:
    print("Optuna = False")
    file = "/saved_models/default_Ant"
    model = A2C('MlpPolicy', env, 
          gamma = 0.99,
          n_steps = 5,
          vf_coef = 0.5,
          ent_coef = 0.0,
          max_grad_norm = 0.5,
          learning_rate = 0.0007,
          gae_lambda = 1.0,       
          rms_prop_eps = 1e-5,
          verbose=0
          )
  # Train the agent
  model.learn(total_timesteps=TOTAL_TIMESTEPS)
  model.save(os.getcwd() + file)
  env.close()

train_model(False)
train_model(True)
