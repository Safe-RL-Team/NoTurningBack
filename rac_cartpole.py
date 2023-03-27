import math
import numpy as np
import os

import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt

from reversibility.offline_reversibility import learn_rev_classifier, learn_rev_action

from stable_baselines3_copy import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env.vec_safe import VecSafe
from stable_baselines3_copy.common.vec_env import DummyVecEnv


model_act = None
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

log_dir = "results/CartpoleRAC"

n_traj_classifier = 10 ** 5
dataset_classifier = 3e5  # they use 3e6
epoch_classifier = 100
no_cuda = False
verbose = True
lr = 0.01
lr_classifier = 0.01
steps_action_model = 10 ** 5
max_steps = 10000
gamma = 0.99

os.makedirs(log_dir, exist_ok=True)

if model_act is None:
    print("Training psi")
    model, buffer = learn_rev_classifier(n_traj=n_traj_classifier,
                                         dataset_size=dataset_classifier,
                                         epochs=epoch_classifier,
                                         lr=lr_classifier,
                                         no_cuda=no_cuda,
                                         verbose=verbose)

    print("Done!")
    print("Training phi")
    model_act = learn_rev_action(model=model,
                                 buffer=buffer,
                                 epochs=steps_action_model,
                                 lr=lr,
                                 no_cuda=no_cuda,
                                 verbose=verbose)
    print("Done!")

    torch.save(model_act, os.path.join(log_dir, 'model_act.pt'))
    torch.save(model, os.path.join(log_dir, 'model_rev.pt'))
else:
    raise NotImplementedError

model_act.device = "cuda"

env = gym.make('CartPole-v0')
env.seed(seed)
env._max_episode_steps = max_steps
env = Monitor(env, os.path.join(log_dir, 'exp_{}'.format(seed)))

env = DummyVecEnv([lambda: env])

thresh_csv = {}
for p_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    episodes_length = []
    if 0 < p_thresh < 1:
        threshold = math.log(p_thresh / (1 - p_thresh))
    else:
        threshold = p_thresh
    env = VecSafe(env, model_act, threshold=threshold)

    for i in range(5):
        episode_length = 0
        obs = env.envs[0].reset()
        while True:
            actions = np.array([env.envs[0].action_space.sample()])
            with torch.no_grad():
                rev_score = model_act(torch.from_numpy(obs).to(model_act.device)).expand(1, -1)
            irrev_idx = rev_score[:, actions].squeeze(1) > threshold
            if irrev_idx.sum() > 0:
                actions[irrev_idx.cpu().numpy()] = torch.argmin(rev_score[irrev_idx], axis=1).cpu().numpy()
            else:
                actions = np.array([torch.argmin(rev_score[0]).cpu().numpy()])
            real_actions = actions

            obs, reward, done, info = env.envs[0].step(np.random.choice(real_actions))
            episode_length += 1
            if done:
                episodes_length.append(episode_length)
                break

    thresh_csv[str(p_thresh)] = episodes_length
    print(p_thresh, ': ', np.mean(episodes_length))

data = pd.DataFrame(thresh_csv)
data.to_csv(log_dir + '/data.csv')
"""
thresh_csv = {}
for p_thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    episodes_length = []
    if 0 < p_thresh < 1:
        threshold = math.log(p_thresh / (1 - p_thresh))
    else:
        threshold = p_thresh
    env = VecSafe(env, model_act, threshold=threshold)

    for i in range(5):
        episode_length = 0
        obs = env.envs[0].reset()
        while True:
            actions = np.array([env.envs[0].action_space.sample()])
            with torch.no_grad():
                rev_score = model_act(torch.from_numpy(obs).to(model_act.device)).expand(1, -1)
            irrev_idx = rev_score[:, actions].squeeze(1) < threshold
            if irrev_idx.sum() > 0:
                actions[irrev_idx.cpu().numpy()] = torch.argmax(rev_score[irrev_idx], axis=1).cpu().numpy()
            real_actions = actions

            obs, reward, done, info = env.envs[0].step(np.random.choice(real_actions))
            episode_length += 1
            if done:
                episodes_length.append(episode_length)
                break

    thresh_csv[str(p_thresh)] = episodes_length
    print(p_thresh, ': ', np.mean(episodes_length))

data = pd.DataFrame(thresh_csv)
data.to_csv(log_dir + '/reverse_data.csv')
"""