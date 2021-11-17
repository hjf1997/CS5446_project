import torch

# env_name = 'CartPole-v1'
env_name = 'LunarLander-v2'
# env_name = 'MountainCar-v0'
model_name = 'DoubleDQ'

gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 100000
# initial_exploration = 32
goal_score = 3000
log_interval = 10
update_target = 100
replay_memory_capacity = 100000
# replay_memory_capacity = 32
save_interval = 10
episodes = 1000
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# Per
small_epsilon = 0.0001
alpha = 0.5
beta_start = 0.1


#Atari
if env_name == 'Breakout-v4':
    replay_memory_capacity = 90000
    update_target = 10000
    initial_exploration = 40000
    log_interval = 100

if env_name == 'MountainCar-v0':
    initial_exploration = 100000
    replay_memory_capacity = 100000
    episodes = 3500
    goal_score = -200
    save_interval = 100
    log_interval = 10

if env_name == 'LunarLander-v2':
    episodes = 3500
    initial_exploration = 10000
    replay_memory_capacity = 10000

if env_name == 'CartPole-v1':
    initial_exploration = 1000
    replay_memory_capacity = 1000
    episodes = 1000

