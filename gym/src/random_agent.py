from craft import Craft
from random import Random
import pickle

import numpy as np

def generate_dataset():
    seed = 2022
    rng = Random(seed)
    env = Craft("./maps/fourobjects.txt", rng)
    episodes = 10000
    episode_len = 150
    dataset = []
    for _ in range(episodes):
        tmp = {'observations': [], 'actions': [], 'rewards': [], 'dones': []}
        env.reset()
        tot_reward = 0
        for t in range(episode_len):
            s0 = env.state
            a = env.rng.randint(0, env.num_actions-1)
            s1, reward, done, info = env.step(a)
            tot_reward += reward
            tmp['observations'].append([s0.uid[0], s0.uid[1]] + [int(elem) for elem in s0.uid[2]])
            tmp['actions'].append(a)
            tmp['rewards'].append(reward)
            tmp['dones'].append(done)
            if done:
                print(tot_reward, "*")
                print(t,"observations", [s0.uid[0], s0.uid[1]] + [int(elem) for elem in s0.uid[2]], "next", s1, "actions", a, "rewards", reward, "dones", done)

        tmp['observations'] = np.array(tmp['observations'])
        tmp['actions'] = np.array(tmp['actions'])
        tmp['rewards'] = np.array(tmp['rewards'])
        tmp['dones'] = np.array(tmp['dones'])
        dataset.append(tmp)
    with open("craft-easy-v1.pkl", 'wb') as handle:
        pickle.dump(dataset, handle)

generate_dataset()