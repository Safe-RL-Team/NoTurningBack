from torchbeast.monobeast import main as impala
from torchbeast.monobeast import parser

# env = gym.make('Sokoban-v0')
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=1000)

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(int(action))
#     env.render()

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.env = 'Sokoban-v0'
    flags.savedir = 'results/SokobanRAE'
    flags.total_steps = 600000

    impala(flags)
    flags.mode = 'test_render'
    impala(flags)
