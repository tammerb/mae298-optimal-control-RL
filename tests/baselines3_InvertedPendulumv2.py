import gym

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('InvertedPendulum-v2')

# Instantiate and train the agent
# model = A2C('MlpPolicy', env, 
#         learning_rate = 7e-4,
#         n_steps = 5,
#         gamma = 0.99,
#         gae_lambda = 1.0,
#         ent_coef = 0.0,
#         vf_coef = 0.5,
#         max_grad_norm = 0.5,
#         rms_prop_eps = 1e-5,
#         verbose=1
#         ).learn(10000)
model = PPO('MlpPolicy', env,
        learning_rate = 3e-4,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.2,
        clip_range_vf = None,
        ent_coef = 0.0,
        vf_coef = 0.5,
        max_grad_norm = 0.5,
        verbose=1
        ).learn(40000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# Render the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()