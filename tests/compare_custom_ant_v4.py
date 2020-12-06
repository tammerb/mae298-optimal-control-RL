import gym
import os
import custom_ant
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 2e5

# Create environment
env = gym.make('CustomAnt-v4')

def train_model(optuna):
  if optuna:
    print("Optuna = True")
    file = "/saved_models/optuna_Custom_Ant_v4"
    model = A2C('MlpPolicy', env, 
          gamma = 0.9998395217363757,
          n_steps = 16,
          vf_coef = 0.5,
          ent_coef = 1.1169937708254059e-05,
          max_grad_norm = 0.9345694121324499,
          learning_rate = 8.208284881543706e-06,
          gae_lambda = 0.9963456366387801,       
          rms_prop_eps = 1e-5,
          verbose=1
          )
  else:
    print("Optuna = False")
    file = "/saved_models/default_Custom_Ant"
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
  #mean_reward, std_reward = eval(model)
  env.close()
  #return mean_reward, std_reward

# Evaluate the trained agent
def eval(model):
  print("Evaluating the model")
  return evaluate_policy(model, env, n_eval_episodes=1)

def print_rewards(mean_reward, std_reward):
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
  

#mean_reward1, std_reward1 = train_model(True)
#mean_reward2, std_reward2 = train_model(False)
train_model(True)

#print_rewards(mean_reward1, std_reward1)
#print_rewards(mean_reward2, std_reward2)
