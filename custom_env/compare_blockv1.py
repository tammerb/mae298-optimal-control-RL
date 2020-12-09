
import gym
import os
import custom_ant
import bestcallback
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback


TOTAL_TIMESTEPS = 2e6

env_name='Block-v1'
models_dir = 'saved_models/'

os.makedirs(models_dir, exist_ok=True)

# Create environment
# env = gym.make(env_name)

def train_model(optuna):
  env = gym.make(env_name)
  
  if optuna:
    print("Optuna = True")
    prefix = "optuna_"
    log_dir = prefix + '/'
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    file = models_dir + prefix + env_name
    # env, callback = setup_callback(log_dir, env)
    model = A2C('MlpPolicy', env, 
          gamma = 0.993630753740229,
          n_steps = 32,
          vf_coef = 0.5,
          ent_coef = 0.16535803309516242,
          max_grad_norm = 0.9345694121324499,
          learning_rate = 0.00021258581917570237,
          gae_lambda = 0.9973243722326772,       
          rms_prop_eps = 1e-5,
          verbose=1
          )
  else:
    print("Optuna = False")
    prefix = "default_"
    log_dir = prefix + '/'
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    file = models_dir + prefix + env_name
    # env, callback = setup_callback(log_dir, env)
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
  
  cb = bestcallback.BestCallback(check_freq=1000, log_dir=log_dir)
 
  # Train the agent
  model.learn(total_timesteps=int(TOTAL_TIMESTEPS), callback=cb)
  model.save(os.getcwd() + file)
  
  mean_reward, std_reward = eval(model)
  env.close()
  return mean_reward, std_reward

# Evaluate the trained agent
def eval(model):
  print("Evaluating the model")
  return evaluate_policy(model, env, n_eval_episodes=1000)

#def print_rewards(mean_reward, std_reward):
#  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
  
def setup_callback(log_dir, env):
  os.makedirs(log_dir, exist_ok=True)
  env = Monitor(env, log_dir)
  callback = BlockCallback(check_freq=1000, log_dir=log_dir)
  return env, callback


mean_reward1, std_reward1 = train_model(True)
mean_reward2, std_reward2 = train_model(False)

print_rewards(mean_reward1, std_reward1)
print_rewards(mean_reward2, std_reward2)

results_plotter.plot_results(["/optuna_"], TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, "Block_v1")
plt.show()