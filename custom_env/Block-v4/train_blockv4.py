
import gym
import os
import custom_ant
from PlotCallBack import PlotCallBack
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines.bench import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback

## User Parameters ##
ENV_ID='Block-v4'
TOTAL_TIMESTEPS = 5.2e5
TRAIN_MODE='BOTH'   # Choose from OPTUNA, DEFAULT, or BOTH
EVALUATE = True     # False will skip the evaluation step
#####################


bag_dir = 'bag/'
os.makedirs(bag_dir, exist_ok=True)
scores = []
results = []

def train_model(optuna, env, bag_dir):
  
  if optuna:
    print("Training with Optuna-optimized hyperparameters")
    prefix = "optuna/"
    log_dir = bag_dir + prefix
    env, callback = setup_callback(log_dir, env)
    model = A2C('MlpPolicy', env, 
          gamma = 0.99987381915872035,
          n_steps = 2,
          ent_coef = 4.948364317173779e-06,
          max_grad_norm = 2.6988398014062644,
          learning_rate = 3.3706472170435117e-06,
          gae_lambda = 0.992596480503565,       
          verbose=0,
          
          )
  else:
    print("Training with stable-baselines3 default hyperparameters")
    prefix = "default/"
    log_dir = bag_dir + prefix
    env, callback = setup_callback(log_dir, env)
    model = A2C('MlpPolicy', env, 
          gamma = 0.99,
          n_steps = 5,
          ent_coef = 0.0,
          max_grad_norm = 0.5,
          learning_rate = 0.0007,
          gae_lambda = 1.0,
          verbose=0
    )
   
  # Train the agent
  model.learn(total_timesteps=int(TOTAL_TIMESTEPS), callback=callback)
  best_model = A2C.load(os.getcwd() + "/" + log_dir + "/best_model", verbose=1)
  
  if EVALUATE: eval(best_model, prefix)
  env.close()
  return

# Evaluate the trained agent
def eval(model, prefix):
  print("Evaluating the best " + prefix + " model")
  mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
  scores.append(prefix + ": mean_reward = " + str(mean_reward) + " +/- " + str(std_reward))

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

legend_names = []

if TRAIN_MODE != 'OPTUNA':
  results.append(bag_dir + "default/")
  env = gym.make(ENV_ID)
  train_model(False, env, bag_dir)
  legend_names.append('Default')
if TRAIN_MODE != 'DEFAULT':
  results.append(bag_dir + "optuna/")  
  env = gym.make(ENV_ID)
  train_model(True, env, bag_dir)
  legend_names.append('Optuna')

for score in scores:
  print(score)

results_plotter.plot_results(results, TOTAL_TIMESTEPS, results_plotter.X_TIMESTEPS, ENV_ID)
plt.legend(legend_names)
plt.show()
