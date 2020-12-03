import gym
import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 5e2

# Create environment
env = gym.make('Ant-v2')

def train_model(optuna):
  if optuna:
    print("Optuna = True")
    file = "/saved_models/optuna_Ant"
    model = A2C('MlpPolicy', env, 
          gamma = 0.9768643682645313,
          n_steps = 64,
          vf_coef = 0.5,
          ent_coef = 3.964131731063129e-06,
          max_grad_norm = 2.0054957616950406,
          learning_rate = 3.762321257388784e-05,
          gae_lambda = 0.925420172167672,       
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
  eval(model)
  env.close()

# Evaluate the trained agent
def eval(model):
  print("Evaluating the model")
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

train_model(True)
train_model(False)
