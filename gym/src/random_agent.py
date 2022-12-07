import random

from craft import Craft, OBJECTS
from random import Random
import pickle
import itertools
import numpy as np
dataset = []

def generate_qlearning_dataset():
    tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
    with open('test3.txt', 'r') as file:
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
    with open("craft-two-v1.pkl", 'wb') as handle:
        pickle.dump(dataset, handle)

def generate_dataset():
    seed = 2022
    rng = Random(seed)
    cnt = 0
    all_permutations = list(itertools.permutations([i for i in range(len(OBJECTS))]))
    for p in all_permutations:
        graph = np.zeros([len(OBJECTS), len(OBJECTS)])
        for i in range(len(p) - 1):
            graph[p[i]][p[i + 1]] = 1
        print(p)
        print(graph)
        env = Craft("./maps/fourobjects.txt", rng, graph)
        episodes = 10000
        episode_len = 80
        for _ in range(episodes):
            tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
            env.reset()
            tot_reward = 0
            for t in range(episode_len):
                s0 = env.state
                cnt += 1
                one_hot_state = env.get_one_hot_state()
                a = env.rng.randint(0, env.num_actions-1)
                s1, reward, done, info = env.step(a)
                tot_reward += reward
                tmp['observations'].append(one_hot_state)
                one_hot_action = np.zeros(env.num_actions)
                one_hot_action[a] = 1
                tmp['actions'].append(one_hot_action)
                tmp['rewards'].append(reward)
                tmp['dones'].append(done)
                if done:
                   # print(tot_reward, "*")
                    print(t,"observations", [s0.uid[0], s0.uid[1]] + [int(elem) for elem in s0.uid[2]], "next", s1, "actions", a, "rewards", reward, "dones", done)
                    break
            tmp['observations'] = np.array(tmp['observations'])
            tmp['actions'] = np.array(tmp['actions'])
            tmp['rewards'] = np.array(tmp['rewards'])
            tmp['dones'] = np.array(tmp['dones'])
            dataset.append(tmp)
    #with open("craft-three-v3.pkl", 'wb') as handle:
       # pickle.dump(dataset, handle)
    print(cnt, "##############")
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


def generate_qlearning_one_hot_dataset():
    tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
    seed = 2022
    rng = Random(seed)
    env = Craft("./maps/fourobjects.txt", rng)
    with open('test3.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            chars = l.strip().split(' ')
            if len(chars) == 3:
                continue
            if len(chars) == 1:
                tmp['observations'] = np.array(tmp['observations'])
                tmp['actions'] = np.array(tmp['actions'])
                tmp['rewards'] = np.array(tmp['rewards'])
                tmp['dones'] = np.array(tmp['dones'])
                dataset.append(tmp)
                tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
                continue

            state_dim= env.observation_space().shape[0]
            tmptmp = []
            for i in range(state_dim):
                tmptmp.append(int(float(chars[i])))
            tmp['observations'].append(tmptmp)

            one_hot_action = np.zeros(env.num_actions)
            one_hot_action[int(chars[state_dim])] = 1
            tmp['actions'].append(one_hot_action)
            tmp['rewards'].append(int(chars[state_dim + 1]))
            if chars[state_dim + 2] == "True":
                tmp['dones'].append(True)
            else:
                tmp['dones'].append(False)
    print(dataset[1]['actions'][1].shape)


   # with open("craft-two-v1.pkl", 'wb') as handle:
       # pickle.dump(dataset, handle)


#generate_qlearning_one_hot_dataset()
generate_dataset()
print(len(dataset[-1]['observations'][-1]))
random.shuffle(dataset)
with open("craft-fourrandom-v1.pkl", 'wb') as handle:
    pickle.dump(dataset, handle)
#test()
