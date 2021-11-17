import matplotlib.pyplot as plt
import numpy as np
import os
from config import env_name, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr, model_name, beta_start, save_interval, episodes


def main():
    root_path = os.path.join('results', env_name)
    # folder_name = ['Q', 'Q_no_reply', 'Q_no_target', 'DoubleDQ', 'DuelDQ', 'Per']
    # legend_name = ['DQN', 'DQN_no_replay', 'DQN_no_target', 'DoubleDQN', 'DuelingDQN', 'PrioritizedDQN']
    folder_name = ['Q', 'DoubleDQ', 'DuelDQ', 'Per']
    legend_name = ['DQN', 'DoubleDQN', 'DuelingDQN', 'PrioritizedDQN']
    file_name = 'score.npy'

    scores = []
    for i in range(len(folder_name)):
        scores.append(np.load(os.path.join(root_path, folder_name[i], file_name)))
        for j in range(1, len(scores[i])):
            scores[i][j] = scores[i][j-1] * 0.99 + scores[i][j] * 0.01
        scores[i] = scores[i][:350]

    # episodes = np.load(os.path.join(root_path, folder_name[0], 'episodes.npy'))
    episodes = np.arange(0, 3500, 10)
    for i in range(len(folder_name)):
        plt.plot(episodes, scores[i], label=legend_name[i])
    plt.legend(loc="upper left")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.grid()
    plt.savefig('LunarLander-v2.jpg')
    plt.show()

if __name__ == '__main__':
    main()