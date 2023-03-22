import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gym_turf import TurfEnv
from stable_baselines3_copy import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env.vec_intrisinc_reward import VecIntrinsic
import torch
from torch.nn import Linear

from reversibility.model import ExtractorGrassland

threshold = 0.8
train_freq = 500
log_dir = "results/turfRAE"
step_penalty = 0
seed = 42
ent_coef = 0.05
lr = 0.01
big = True

gradient_step = 1
learning_start = 0
batch_size = 128
buffer_size = 10 ** 6
d_min = 0
d_max = 50000
reward_free = False
time_steps = 5e5
max_steps = 480

use_gpu = torch.cuda.is_available()
wir = 0.1

np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs(log_dir, exist_ok=True)


def func(x):
    return (x > threshold) * (x - threshold)


env = TurfEnv(step_penalty=step_penalty, big=big, max_steps=max_steps)
env = Monitor(env, os.path.join(log_dir, 'exp'), info_keywords=('ruined grasses',))
env.seed(seed)

model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, clip_range_vf=None, ent_coef=ent_coef)

head = Linear(2 * 64, 1)
extractor = ExtractorGrassland(model.env.observation_space, features_dim=64)
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

data = pd.DataFrame({'rewards': env.all_rewards,
                     'spoiled_grass': env.all_spoiled_grass})
data.to_csv(log_dir + '/data.csv.csv')
np.savetxt(log_dir + '/all_positions.txt', env.all_positions, fmt='%d')

plt.plot(env.all_rewards)
plt.show()
plt.plot(env.all_spoiled_grass)
plt.show()
print(env.all_positions)
