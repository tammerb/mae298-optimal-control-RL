import gym
import os

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

TOTAL_TIMESTEPS = 10000
NUM_EVAL_EPIS = 1000

# Create environment
env = gym.make('CartPole-v1')

# Instantiate the agent
model1 = A2C('MlpPolicy', env, 
        learning_rate = 7e-4,
        n_steps = 5,
        gamma = 0.99,
        gae_lambda = 1.0,
        ent_coef = 0.0,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        rms_prop_eps = 1e-5,
        verbose=0
        )

model2 = A2C('MlpPolicy', env, 
        learning_rate = 0.0015613269950806646,
        n_steps = 256,
        gamma = 0.9987940879356862,
        gae_lambda = 0.994637205730802,
        ent_coef = 1.2954655366837278e-06,
        vf_coef = 0.5,
        max_grad_norm = 1.92012534215678,
        rms_prop_eps = 1e-5,
        verbose=0
        )

# Train the agent
print("Training default model")
model1.learn(total_timesteps=TOTAL_TIMESTEPS)
model1.save(os.getcwd() + "/default_cartpole")
print("Training optuna model")
model2.learn(total_timesteps=TOTAL_TIMESTEPS)
model2.save(os.getcwd() + "/optuna_cartpole")

# Evaluate the trained agent
print("Evaluating the models")
mean_reward1, std_reward1 = evaluate_policy(model1, env, n_eval_episodes=NUM_EVAL_EPIS, deterministic=True)
mean_reward2, std_reward2 = evaluate_policy(model2, env, n_eval_episodes=NUM_EVAL_EPIS, deterministic=True)
print(f"Default model: mean_reward={mean_reward1:.2f} +/- {std_reward1}")
print(f"Optuna model: mean_reward={mean_reward2:.2f} +/- {std_reward2}")

# Render the trained agent
obs = env.reset()
for i in range(500):
    action, _states = model1.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
obs = env.reset()
for i in range(500):
    action, _states = model2.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()