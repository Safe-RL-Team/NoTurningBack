import os

import matplotlib.pyplot as plt
import numpy as np
import gym
import pandas as pd

from stable_baselines3_copy import PPO
from stable_baselines3_copy.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env.vec_intrisinc_reward import VecIntrinsic
import torch
from torch.nn import Linear

from reversibility.model import ExtractorCartpole

threshold = 0.7
train_freq = 500
log_dir = "results/CartpoleRAE"
step_penalty = 0
seed = 42
lr = 0.01
ent_coef = 0.05

gradient_step = 1
learning_start = 0
batch_size = 128
buffer_size = 10 ** 6
d_min = 0
d_max = 50000
reward_free = True
time_steps = 500000


use_gpu = torch.cuda.is_available()
wir = 1

np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs(log_dir, exist_ok=True)


def func(x):
    return (x > threshold) * (x - threshold)


env = gym.make('CartPole-v0')
env = Monitor(env, os.path.join(log_dir, 'exp'))
env.seed(seed)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir, clip_range_vf=None, ent_coef=ent_coef)

head = Linear(2 * 64, 1)
extractor = ExtractorCartpole()
if use_gpu:
    head = head.cuda()
    extractor = extractor.cuda()
head.train()
extractor.train()

model.env = VecIntrinsic(model.env, feature_extractor=extractor, head=head,
                         weight_intrinsic_reward=wir,
                         func=func,
                         lr=lr,
                         train_freq=train_freq,
                         gradient_step=gradient_step,
                         learning_start=learning_start,
                         batch_size=batch_size,
                         buffer_size=buffer_size,
                         d_min=d_min,
                         d_max=d_max,
                         reward_free=reward_free,
                         save_path=log_dir
                         )

model.learn(total_timesteps=time_steps)

model.save(os.path.join(log_dir, 'model.pt'))

length = np.array(model.env.episode_length_recording)
reward = np.array(model.env.reward_recording).flatten()[:len(length)]

data = pd.DataFrame({'length': length, 'intrins_rew': reward})
data.to_csv(log_dir + '/data.csv')

plt.plot(length)
plt.show()
plt.plot(reward)
plt.show()
