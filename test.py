import os
import sys
import gym
import random
import numpy as np
from gif import save_frames_as_gif
import torch
import torch.optim as optim
import torch.nn.functional as F
from model import QNet, DuelDQNet, DoubleDQNet, PerNet

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, model_name
from gym_wrappers import MainGymWrapper

def get_action(state, online_net, env):
    return online_net.get_action(state)


def main(e):
    env = gym.make(env_name)

    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    model = eval(model_name + 'Net')
    online_net = model(num_inputs, num_actions)
    print('load from:' + os.path.join('checkpoint', env_name, model_name, model_name + '_' + str(e) + '.pth'))
    online_net.load_state_dict(torch.load(os.path.join('checkpoint', env_name, model_name, model_name + '_' + str(e) + '.pth')))
    # online_net.load_state_dict(torch.load(os.path.join('./checkpoint', 'LunarLander-v2', 'dqn_no_target.pth')))
    online_net.to(device)

    # test
    frames = []
    score_games = 0
    for i in range(10):
        done = False
        score = 0
        state = env.reset()
        state = torch.Tensor(np.asarray(state).astype(np.float)).to(device)
        state = state.unsqueeze(0)
        while not done:
            frames.append((env.render(mode="rgb_array"), i, score))
            env.render()
            action = get_action(state, online_net, env)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(np.asarray(next_state).astype(np.float64))
            next_state = next_state.unsqueeze(0)

            action_one_hot = np.zeros(num_actions)
            action_one_hot[action] = 1

            score += reward
            state = next_state
            print(done)
            print(score)
        score_games += score
    print('avg Score: ' + str(score_games / 10))
    save_frames_as_gif(frames)

if __name__=="__main__":
    e = 3490
    main(e)