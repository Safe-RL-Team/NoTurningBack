import math
import numpy as np
import os
import torch

from reversibility.offline_reversibility import learn_rev_classifier, learn_rev_action

from stable_baselines3_copy import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env.vec_safe import VecSafe
from stable_baselines3_copy.common.vec_env import DummyVecEnv

from gym_turf import TurfEnv


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

p_thresh = 0.7

log_dir = "results/turfRAC"

n_traj_classifier = 10 ** 4
dataset_classifier = 10 ** 2
epoch_classifier = 10 ** 2
no_cuda = False
verbose = True
lr_classifier = 0.01
steps_action_model = 10 ** 2
lr_classifier_act = 0.01
step_penalty = 0
ent_coef = 0.05
time_steps = 10 ** 3

os.makedirs(log_dir, exist_ok=True)

print("Training psi")
model, buffer = learn_rev_classifier(n_traj=n_traj_classifier,
                                     env_str='turf',
                                     dataset_size=dataset_classifier,
                                     epochs=epoch_classifier,
                                     lr=lr_classifier,
                                     no_cuda=no_cuda,
                                     verbose=verbose)

print("Done!")
print("Training phi")
model_act = learn_rev_action(model=model,
                             env_str='turf',
                             buffer=buffer,
                             epochs=steps_action_model,
                             lr=lr_classifier_act,
                             no_cuda=no_cuda,
                             verbose=verbose)
print("Done!")

torch.save(model_act, os.path.join(log_dir, 'model_act.pt'))
torch.save(model, os.path.join(log_dir, 'model_rev.pt'))

model_act.device = "cuda"

env = TurfEnv(step_penalty=step_penalty)
env = Monitor(env, os.path.join(log_dir, 'exp'), info_keywords=('ruined grasses',))
env.seed(seed)

env = DummyVecEnv([lambda: env])


model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=log_dir, clip_range_vf=None, ent_coef=ent_coef)


if p_thresh < 1:
    threshold = math.log(p_thresh / (1 - p_thresh))
    model.env = VecSafe(model.env, model_act, threshold=threshold)

model.learn(total_timesteps=time_steps)

model.save(os.path.join(log_dir, 'model.pt'))
