import os
import gym
import custom_ant

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

optuna = False

if optuna:
    print("Optuna = True")
    file = "/saved_models/optuna_InvertedPendulum"
else:
    print("Optuna = False")
    file = "/saved_models/default_InvertedPendulum"

model = A2C.load(os.getcwd() + file, verbose=1)
env = gym.make('InvertedPendulum-v2')

# Evaluate the trained agent
print("Evaluating the model")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Render the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()