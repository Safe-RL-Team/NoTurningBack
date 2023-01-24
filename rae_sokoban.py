import gym
import torch
import gym_sokoban  # https://github.com/mpSchrader/gym-sokoban

import numpy as np

from stable_baselines3 import PPO

from torchbeast.monobeast import main as impala
from torchbeast.monobeast import parser

# env = gym.make('Sokoban-v0')

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.env = 'Sokoban-v0'
    impala(flags)
    flags.mode = 'test_render'
    impala(flags)

# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=1000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(int(action))
#     env.render()
