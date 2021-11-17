import gym
import numpy as np
import os

import torch
import torch.optim as optim
from atari_model import QNet, DoubleDQNet, DuelDQNet, PerNet
from memory import Memory, Memory_With_TDError

from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, model_name, beta_start, save_interval
from gym_wrappers import MainGymWrapper


def get_action(state, target_net, epsilon, env):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        action = target_net.get_action(state)
        return action

def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


def main():
    env = MainGymWrapper.wrap(gym.make(env_name))
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    model = eval(model_name + 'Net')
    online_net = model(num_inputs, num_actions)
    target_net = model(num_inputs, num_actions)
    update_target_model(online_net, target_net)

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity) if model_name != 'per' else Memory_With_TDError(replay_memory_capacity)
    running_score = 0
    epsilon = 1.0
    steps = 0
    beta = beta_start
    loss = 0
    scores_save = []
    episodes_save = []

    for e in range(10000):
        done = False

        score = 0
        state = env.reset()
        state = torch.Tensor(np.asarray(state).astype(np.float)).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = get_action(state, target_net, epsilon, env)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(np.asarray(next_state).astype(np.float)).to(device)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            action_one_hot = np.zeros(num_actions)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            if steps > initial_exploration:
                epsilon -= 0.000003
                epsilon = max(epsilon, 0.1)

                if model_name != 'per':
                    batch = memory.sample(batch_size)
                    loss = model.train_model(online_net, target_net, optimizer, batch, device)
                else:
                    beta += 0.00005
                    beta = min(1, beta)
                    batch, weights = memory.sample(batch_size, online_net, target_net, beta)
                    loss = PerNet.train_model(online_net, target_net, optimizer, batch, weights)

                if steps % update_target == 0:
                    update_target_model(online_net, target_net)

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))

        if e % save_interval == 0:
            scores_save.append(running_score)
            episodes_save.append(e)
            # save model
            online_net.to(torch.device('cpu'))
            torch.save(online_net.state_dict(), 'checkpoint/' + model_name + '.pth')
            online_net.to(device)
            # save scores
            if not os.path.isdir(os.path.join('results', env_name, model_name)):
                os.mkdir(os.path.join('results', env_name, model_name))
            np.save(os.path.join('results', env_name, model_name, 'score.npy'), np.array(scores_save))
            np.save(os.path.join('results', env_name, model_name, 'episodes.npy'), np.array(episodes_save))

        if running_score > goal_score:
            break


if __name__=="__main__":
    main()
