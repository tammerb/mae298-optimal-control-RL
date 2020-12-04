import gym
import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 5e6

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
          verbose=1
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
          verbose=1
          )
  # Train the agent
  model.learn(total_timesteps=TOTAL_TIMESTEPS)
  model.save(os.getcwd() + file)
  mean_reward, std_reward = eval(model)
  env.close()
  return mean_reward, std_reward

# Evaluate the trained agent
def eval(model):
  print("Evaluating the model")
  return evaluate_policy(model, env, n_eval_episodes=100)

def print_rewards(mean_reward, std_reward):
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
  

mean_reward1, std_reward1 = train_model(True)
mean_reward2, std_reward2 = train_model(False)

print_rewards(mean_reward1, std_reward1)
print_rewards(mean_reward2, std_reward2)
