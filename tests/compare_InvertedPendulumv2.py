
import gym
import os
from PlotCallBack import PlotCallBack
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines.bench import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback





TOTAL_TIMESTEPS = 5e5
env_name='InvertedPendulum-v2'

bag_dir = env_name + '_bag/'
os.makedirs(bag_dir, exist_ok=True)

def train_model(optuna, env, bag_dir):
  
  if optuna:
    print("Optuna = True")
    prefix = "optuna/"
    log_dir = bag_dir + prefix
    model_dir = bag_dir + prefix
    env, callback = setup_callback(log_dir, env)
    model = A2C('MlpPolicy', env, 
          gamma = 0.9926635896428226,
          n_steps = 32,
          vf_coef = 0.5,
          ent_coef = 1.9546965597732253e-08,
          max_grad_norm = 1.7716329511301456,
          learning_rate = 0.0010397208127972074,
          gae_lambda = 0.9264481442403701,       
          rms_prop_eps = 1e-5,
          verbose=0
          )
  else:
    print("Optuna = False")
    prefix = "default/"
    log_dir = bag_dir + prefix
    model_dir = bag_dir + prefix
    env, callback = setup_callback(log_dir, env)
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
  model.learn(total_timesteps=int(TOTAL_TIMESTEPS), callback=callback)
  model.save(os.getcwd() + '/' + model_dir)
  
  mean_reward, std_reward = eval(model)
  env.close()
  return mean_reward, std_reward

# Evaluate the trained agent
def eval(model):
  print("Evaluating the model")
  return evaluate_policy(model, env, n_eval_episodes=10)

def print_rewards(mean_reward, std_reward):
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
  
def setup_callback(log_dir, env):
  if os.path.isfile(log_dir + 'monitor.csv'):
    print("A monitor.csv already exists. Deleting it.")
    os.remove(log_dir + 'monitor.csv')
    if not os.path.isfile(log_dir + 'monitor.csv'):
      print("Old monitor.csv successfully deleted")
  os.makedirs(log_dir, exist_ok=True)
  env = Monitor(env, log_dir)
  callback = PlotCallBack(check_freq=1000, log_dir=log_dir)
  return env, callback

env = gym.make(env_name)
mean_reward1, std_reward1 = train_model(False, env, bag_dir)
env = gym.make(env_name)
mean_reward2, std_reward2 = train_model(True, env, bag_dir)

print_rewards(mean_reward1, std_reward1)
print_rewards(mean_reward2, std_reward2)

results_plotter.plot_results([bag_dir + "default/", bag_dir + "optuna/"], TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, env_name)
plt.show()
