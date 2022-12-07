import sys  # noqa
from os import path as p  # noqa
sys.path.append(p.abspath(p.join(p.dirname(sys.modules[__name__].__file__),
                                 "..")))  # noqa

import logging
from os import path
from random import Random
from time import time

from craft import Craft, CraftState
from rl import Agent, EpsilonGreedy
from rl.utils import SequenceReport

from craft import OBJECTS
from rl.common import ReachFacts

DEFAULT_Q = 0.0
TOTAL_STEPS = 700000
EPISODE_LENGTH = 80
LOG_STEP = 10000
TRIALS = 5

logging.basicConfig(level=logging.INFO)

def print_state(state, action):
    if action == 0:
        print("Agent Location:", state.x, state.y, "Action: Down")
    elif action == 1:
        print("Agent Location:", state.x, state.y, "Action: Up")
    elif action == 2:
        print("Agent Location:", state.x, state.y,  "Action: Left")
    elif action == 3:
        print("Agent Location:", state.x, state.y,  "Action: Right")
    elif action == 4:
        print("Agent Location:", state.x, state.y,  "Action: Do")
    else:
        print("Agent Location:", state.x, state.y)


def evaluate_agent(env, policy1, reward1, init):
    print("Evaluation:")
    state_rewards = []
    for initial_state1 in init:
        env.reset(initial_state1)
        reward1.reset()
        policy1.reset(evaluation=True)

        trial_reward: float = 0.0

        for step in range(EPISODE_LENGTH):
            s0 = env.state
            a = policy1.get_best_action(s0)
            next_state, cur_reward, cur_done, _ = env.step(a)
            s1 = env.state
            print_state(s0, a)
            step_reward, finished = reward1(s0, a, s1)
            if not finished:
                trial_reward += step_reward
            logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)
            if finished:
                print_state(s1, -1)
                break

        state_rewards.append(trial_reward)



def run(filename, seed):

    here = path.dirname(__file__)
    map_fn = "./maps/fourobjects.txt"
    print(map_fn)
    init = [CraftState(1, 1, set())]

    rng = Random(seed)

    env = Craft(map_fn, rng)

    goal = [OBJECTS["wood"], OBJECTS["iron"], OBJECTS["gold"],  OBJECTS["gem"]]

    with open(filename, "w") as csvfile:
        print("ql: begin experiment")
        report = SequenceReport(csvfile, LOG_STEP, init, EPISODE_LENGTH, TRIALS)


        print("begin")
        rng.seed(seed)

        reward = ReachFacts(env, goal)
        policy = EpsilonGreedy(alpha=1.0, gamma=0.99, epsilon=0.1,
                               default_q=DEFAULT_Q, num_actions=5, rng=rng)
        agent = Agent(env, policy, reward, rng)
        try:
            start = time()
            agent.train(steps=TOTAL_STEPS,
                        steps_per_episode=EPISODE_LENGTH, report=report)
            end = time()
            print("ql: Ran task for {} seconds.".format(end - start))
        except KeyboardInterrupt:
            end = time()

        report.increment(TOTAL_STEPS)
        for d in report.dataset:
            for t in d:
                report.writer.writerow(t)
            report.writer.writerow("#")
        evaluate_agent(env, policy, reward, init)

run("./test3.txt", 2022)