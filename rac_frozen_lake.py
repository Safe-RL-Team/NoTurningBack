import math
import numpy as np
import os
import torch
import gym

from reversibility.offline_reversibility import learn_rev_classifier, learn_rev_action

from stable_baselines3_copy import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3_copy.common.vec_env.vec_safe import VecSafe
from stable_baselines3_copy.common.vec_env import DummyVecEnv


model_act = None
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

p_thresh = 0.7

log_dir = "results/FrozenLakeRAC"
slippery_training = False
slippery_testing = False

n_traj_classifier = 10 ** 4
dataset_classifier = 10 ** 2
epoch_classifier = 10 ** 2
no_cuda = False
verbose = True
lr_classifier = 0.01
steps_action_model = 10 ** 2
time_steps = 10 ** 3
max_steps = 200
gamma = 0.99

os.makedirs(log_dir, exist_ok=True)

if model_act is None:
    print("Training psi")
    model, buffer = learn_rev_classifier(n_traj=n_traj_classifier,
                                         env_str='frozenlake',
                                         dataset_size=dataset_classifier,
                                         epochs=epoch_classifier,
                                         lr=lr_classifier,
                                         no_cuda=no_cuda,
                                         verbose=verbose,
                                         slippery=slippery_training)

    print("Done!")
    print("Training phi")
    model_act = learn_rev_action(model=model,
                                 buffer=buffer,
                                 env_str='frozenlake',
                                 epochs=steps_action_model,
                                 lr=lr_classifier,
                                 no_cuda=no_cuda,
                                 verbose=verbose)
    print("Done!")

    torch.save(model_act, os.path.join(log_dir, 'model_act.pt'))
    torch.save(model, os.path.join(log_dir, 'model_rev.pt'))
else:
    raise NotImplementedError

model_act.device = "cuda"

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=slippery_testing)
env.seed(seed)
env._max_episode_steps = max_steps
env = Monitor(env, os.path.join(log_dir, 'exp_{}'.format(seed)))

env = DummyVecEnv([lambda: env])


if p_thresh < 1:
    threshold = math.log(p_thresh / (1 - p_thresh))
    env = VecSafe(env, model_act, threshold=threshold)

model = PPO('MlpPolicy', env, verbose=1,
            tensorboard_log=log_dir,
            gamma=gamma)

model.learn(total_timesteps=time_steps)

model.save(os.path.join(log_dir, 'model.pt'))

print(env.envs[0].episode_returns)
print(env.envs[0].episode_lengths)
