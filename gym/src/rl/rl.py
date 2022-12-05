import logging
from abc import ABC, abstractmethod
from random import Random
from typing import List, Optional, Sequence, Tuple

from .environment import Environment, RewardFn, State
from .utils import Report


class Policy(ABC):
    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_best_action(self, state: State,
                        restrict: Optional[List[int]] = None):
        pass

    @abstractmethod
    def get_train_action(self, state: State,
                         restrict: Optional[List[int]] = None):
        pass

    def report(self) -> str:
        return ""

    @abstractmethod
    def reset(self, evaluation: bool):
        pass

    @abstractmethod
    def update(self, s0: State, a, s1: State, r: float, end: bool):
        pass


class Agent:
    environment: Environment
    policy: Policy
    reward: RewardFn
    report_reward: RewardFn
    rng: Random

    def __init__(self, environment: Environment, policy: Policy,
                 reward: RewardFn, rng: Random, report_diff=False, report_reward: RewardFn = None):
        self.environment = environment
        self.policy = policy
        self.reward = reward
        self.rng = rng
        if report_diff:
            self.report_reward = report_reward
        else:
            self.report_reward = reward

    def demo(self, state: State, steps: int):
        rng_state = self.rng.getstate()

        self.environment.reset(state)
        self.reward.reset()
        self.policy.reset(evaluation=True)

        total_reward: float = 0.0

        print(state)

        for step in range(steps):
            s0 = self.environment.state

            a = self.policy.get_best_action(s0)
            self.environment.apply_action(a)
            print(a)

            s1 = self.environment.state
            print(s1)

            step_reward, finished = self.reward(s0, a, s1)
            print(step_reward)

            total_reward += step_reward

            if finished:
                print("fin")
                break
        print("total reward:", total_reward)

        self.rng.setstate(rng_state)

    def evaluate(self, states: Sequence[State], steps: int, trials: int,
                 name: str = "") -> List[float]:
        state_rewards: List[float] = []
        rng_state = self.rng.getstate()
        for initial_state in states:
            trial_rewards: List[float] = []
            for trial in range(trials):
                self.environment.reset(initial_state)
                self.report_reward.reset()
                self.policy.reset(evaluation=True)
                trial_reward: float = 0.0
                for step in range(steps):
                    s0 = self.environment.state

                    a = self.policy.get_best_action(s0)
                    self.environment.step(a)

                    s1 = self.environment.state

                    step_reward, finished = self.report_reward(s0, a, s1)

                    trial_reward += step_reward

                    logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)

                    if finished:
                        break

                logging.debug("Evaluation from %s obtained reward of %s.",
                              initial_state, trial_reward)

                trial_rewards.append(trial_reward)
            state_rewards.append(sum(trial_rewards)/trials)

        logging.info("Evaluation %s obtained %s average reward per trial. %s",
                     name, sum(state_rewards)/len(states),
                     self.policy.report())

        self.rng.setstate(rng_state)
        return state_rewards

    def train_episode(self, current_step: int, end_step: int,
                      report: Optional[Report] = None) -> Tuple[int, float,
                                                                bool]:
        self.environment.reset()
        self.reward.reset()
        self.policy.reset(evaluation=False)

        episode_reward = 0.0
        finished = False

        logging.debug("Begin episode (current step is %s)", current_step)

        tmp = []
        for step in range(current_step, end_step):
            if finished:
                report.dataset.append(tmp)
                break

            s0 = self.environment.state
            one_hot_state = self.environment.get_one_hot_state()
            a = self.policy.get_train_action(s0)
            cur_n_state, cur_reward, cur_done, _ = self.environment.step(a)
            tmp.append(list(one_hot_state) + [a, cur_reward, cur_done])

            s1 = self.environment.state

            step_reward, finished = self.reward(s0, a, s1)
            self.policy.update(s0, a, s1, step_reward, finished)
            episode_reward += step_reward

            logging.debug("(%s, %s, %s) -> %s", s0, a, s1, step_reward)

            if step % report.log_step == 0:
                self.evaluate(report.states, report.steps, report.trials, name=str(step))
           # if report is not None:
               # report.evaluate(self, step + 1)
        logging.debug("End episode (last step was %s, success=%s)", step,
                      finished)

        return step + 1, episode_reward, finished

    def train(self, steps: int, steps_per_episode: int,
              report: Optional[Report] = None) -> Tuple[float, int, int]:
        total_reward: float = 0.0
        current_step: int = 0
        episode: int = 0

        if report is not None:
            report.evaluate(self, 0, force=True)

        while current_step < steps:
            next_end = min(current_step + steps_per_episode, steps)
            current_step, episode_reward, finished = \
                self.train_episode(current_step, next_end, report)

            total_reward += episode_reward
            logging.debug("Training episode %s obtained reward %s after %s"
                          "steps.", episode, episode_reward, current_step + 1)

            episode += 1

        logging.info("Training finished after %s total steps (%s episodes).",
                     current_step, episode)
        logging.info("Total cumulative reward was %s.", total_reward)

        return total_reward, current_step, episode
