import os
import numpy as np

from gym_turf import TurfEnv
from stable_baselines3_copy import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env.vec_intrisinc_reward import VecIntrinsic
import torch
from torch.nn import Linear

from reversibility.model import ExtractorGrassland

# extract: states during training, reward, spoiled grass
# for cartpole: score, state + estimated reversibility, intrinsic reward etc.

threshold = 0.8
train_freq = 500
log_dir = "results/turfRAE"
step_penalty = 0
seed = 42
ent_coef = 0.05

gradient_step = 10
learning_start = 0
batch_size = 128
buffer_size = 2000
d_min = 0
d_max = 100
reward_free = False
time_steps = 10 ** 4


use_gpu = torch.cuda.is_available()
wir = 0.1

np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs(log_dir, exist_ok=True)


def func(x):
    return (x > threshold) * (x - threshold)


env = TurfEnv(step_penalty=step_penalty)
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
                         )

model.learn(total_timesteps=time_steps)

model.save(os.path.join(log_dir, 'model.pt'))

# list of total reward per episode (0 or 1)
print(env.all_rewards)
# list of total spoiled grass per episode
print(env.all_spoiled_grass)
# array of visits per node
print(env.all_positions)
