import os
import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3_copy import PPO
from stable_baselines3_copy.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env.vec_intrisinc_reward import VecIntrinsic
import torch
from torch.nn import Linear

from reversibility.model import ExtractorFrozenLake

five = [
    "SFFFH",
    "FFHFF",
    "HFFFH",
    "FFHFF",
    "HFFFG"
    ]

ten = [
    "SFFFHFFFHF",
    "FFHFFFHFFF",
    "HFFFHFFFHF",
    "FFHFFFHFFF",
    "HFFFHFFFHF",
    "FFHFFFHFFF",
    "HFFFHFFFHF",
    "FFHFFFHFFF",
    "HFFFHFFFHF",
    "FFHFFFHFFG"
]
suicide_test = [
    "SFFFG",
    "FFFFF",
    "FFFFF",
    "FFFFF",
    "HFFFF"
    ]

threshold = 0.8
train_freq = 500
log_dir = "results/FrozenLakeRAE"
slippery = True

step_penalty = 0
seed = 42
ent_coef = 0.05
lr = 0.01

gradient_step = 1
learning_start = 0
batch_size = 128
buffer_size = 10 ** 6
d_min = 0
d_max = 50000
reward_free = False
time_steps = 5e5


use_gpu = torch.cuda.is_available()
wir = 0.1

np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs(log_dir, exist_ok=True)


def func(x):
    return (x > threshold) * (x - threshold)


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery)
env = Monitor(env, os.path.join(log_dir, 'exp'))
env.seed(seed)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, clip_range_vf=None, ent_coef=ent_coef)

head = Linear(2 * 64, 1)
extractor = ExtractorFrozenLake()
if use_gpu:
    head = head.cuda()
    extractor = extractor.cuda()
head.train()
extractor.train()

model.env = VecIntrinsic(model.env, feature_extractor=extractor, head=head,
                         weight_intrinsic_reward=wir,
                         func=func,
                         train_freq=train_freq,
                         gradient_step=gradient_step,
                         learning_start=learning_start,
                         batch_size=batch_size,
                         buffer_size=buffer_size,
                         d_min=d_min,
                         d_max=d_max,
                         reward_free=reward_free,
                         save_path=log_dir,
                         lr=lr
                         )

model.learn(total_timesteps=time_steps)

model.save(os.path.join(log_dir, 'model.pt'))

positions = np.zeros((4, 4))
for pos in np.array(model.env.obs_recording):
    positions[int(pos / 4), int(pos % 4)] += 1

np.savetxt(log_dir + '/all_positions.txt', positions, fmt='%d')
data = pd.DataFrame({'eps_rewards': env.episode_rewards})
data.to_csv(log_dir + '/data.csv')

print(positions)
plt.plot(env.episode_rewards)
plt.show()
