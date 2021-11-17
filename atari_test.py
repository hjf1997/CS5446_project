import os
import sys
import gym
import random
import numpy as np
from gif import save_frames_as_gif
import torch
import torch.optim as optim
import torch.nn.functional as F
from atari_model import QNet, DuelDQNet, DoubleDQNet, PerNet
from memory import Memory

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr
from gym_wrappers import MainGymWrapper

def get_action(state, online_net, env):
    return online_net.get_action(state)


def main():
    env = MainGymWrapper.wrap(gym.make(env_name, render_mode='human'))

    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    online_net = DoubleDQNet(num_inputs, num_actions)
    online_net.load_state_dict(torch.load('checkpoint/Q.pth'))
    online_net.to(device)

    # test
    done = False
    frames = []
    score = 0
    state = env.reset()
    state = torch.Tensor(np.asarray(state).astype(np.float)).to(device)
    state = state.unsqueeze(0)

    while not done:
        # frames.append(env.render(mode="rgb_array"))
        # env.render()
        action = get_action(state, online_net, env)
        next_state, reward, done, _ = env.step(action)

        next_state = torch.Tensor(np.asarray(next_state).astype(np.float64))
        next_state = next_state.unsqueeze(0)

        reward = reward if not done or score == 499 else -1
        action_one_hot = np.zeros(num_actions)
        action_one_hot[action] = 1

        score += reward
        state = next_state
        print(done)
        print(score)

    # save_frames_as_gif(frames)

if __name__=="__main__":
    main()