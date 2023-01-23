import gym
import gym_sokoban  # https://github.com/mpSchrader/gym-sokoban

import numpy as np

from stable_baselines3 import PPO

env = gym.make('Sokoban-v0')

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(int(action))
    env.render()
