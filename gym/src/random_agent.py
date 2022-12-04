from craft import Craft
from random import Random
import pickle

import numpy as np

def generate_qlearning_dataset():
    dataset = []
    tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
    with open('test2.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            chars = l.strip().split(' ')
            if len(chars) == 1:
                tmp['observations'] = np.array(tmp['observations'])
                tmp['actions'] = np.array(tmp['actions'])
                tmp['rewards'] = np.array(tmp['rewards'])
                tmp['dones'] = np.array(tmp['dones'])
                dataset.append(tmp)
                tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
                continue

            tmp['observations'].append([int(chars[0]), int(chars[1]), int(chars[2]), int(chars[3]), int(chars[4])])
            one_hot_action = np.zeros(5)
            one_hot_action[int(chars[5])] = 1
            tmp['actions'].append(one_hot_action)
            tmp['rewards'].append(int(chars[6]))
            if chars[7] == "True":
                tmp['dones'].append(True)
            else:
                tmp['dones'].append(False)
    print(dataset[1]['actions'][1].shape)
    with open("craft-three-v3.pkl", 'wb') as handle:
        pickle.dump(dataset, handle)

def generate_dataset():
    seed = 2022
    rng = Random(seed)
    env = Craft("./maps/threeobjects.txt", rng)
    episodes = 10000
    episode_len = 50
    dataset = []
    for _ in range(episodes):
        tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
        s0 = env.reset()
        tot_reward = 0
        for t in range(episode_len):
            s0 = env.state
            a = env.rng.randint(0, env.num_actions-1)
            s1, reward, done, info = env.step(a)
            tot_reward += reward
            tmp['observations'].append([s0.uid[0], s0.uid[1]] + [int(elem) for elem in s0.uid[2]])
            one_hot_action = np.zeros(5)
            one_hot_action[a] = 1
            tmp['actions'].append(one_hot_action)
            tmp['rewards'].append(reward)
            tmp['dones'].append(done)
            if done:
                print(tot_reward, "*")
                print(t,"observations", [s0.uid[0], s0.uid[1]] + [int(elem) for elem in s0.uid[2]], "next", s1, "actions", a, "rewards", reward, "dones", done)
                break
        tmp['observations'] = np.array(tmp['observations'])
        tmp['actions'] = np.array(tmp['actions'])
        tmp['rewards'] = np.array(tmp['rewards'])
        tmp['dones'] = np.array(tmp['dones'])
        dataset.append(tmp)
    with open("craft-three-v3.pkl", 'wb') as handle:
        pickle.dump(dataset, handle)

def test():
    dataset_path = '../tfe-default-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)


    print("######", trajectories[10]['observations'][10])
    print("######", trajectories[10]['actions'])
    print("######", trajectories[10]['rewards'])
    print("######", trajectories[10]['terminals'])
    print("####")
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)
    print(returns)
    print(np.argmax(returns), "$$$$$$$$$$$$$$$$$")
    print(returns[np.argmax(returns)])

    print('=' * 50)
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

#generate_qlearning_dataset()
test()