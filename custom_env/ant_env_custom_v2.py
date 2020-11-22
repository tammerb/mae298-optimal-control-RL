import gym
import mujoco_py
import custom_ant


env = gym.make('CustomAnt-v2')

#Sets an initial state
env.reset()
# Rendering our instance 300 times
for _ in range(300):
  #renders the environment
  env.render()
  #Takes a random action from its action space 
  # aka the number of unique actions an agent can perform
  env.step(env.action_space.sample())
env.close()
