from csv import writer
from sys import stdout
from typing import IO, Sequence

from ..environment import State

from ..rl import *


class Report:
    def __init__(self, logfile: IO, log_step: int, states: Sequence[State],
                 steps: int, trials: int):
        self.writer = writer(logfile, delimiter=' ')
        self.log_step = log_step
        self.states = states
        self.steps = steps
        self.trials = trials
        self.dataset = []

    def evaluate(self, agent: 'rl.Agent', step: int, force: bool = False):
        if not force and step % self.log_step != 0:
            return
        values = agent.evaluate(self.states, self.steps, self.trials,
                                name=str(step))

        if len(values) == 0:
            return

        mean = sum(values) / len(values)
        values.sort()

        self.writer.writerow([step, mean] + values)


class StdoutReport(Report):
    def __init__(self, log_step: int, states: Sequence[State], steps: int,
                 trials: int):
        super().__init__(stdout, log_step, states, steps, trials)


class SequenceReport(Report):
    start: int

    def __init__(self, logfile: IO, log_step: int, states: Sequence[State],
                 steps: int, trials: int):
        self.start = 0
        super().__init__(logfile, log_step, states, steps, trials)

    def evaluate(self, agent: 'rl.Agent', step: int, force: bool = False):
        if force or step % self.log_step == 0:
            super().evaluate(agent, step + self.start, force=True)

    def increment(self, steps: int):
        self.start += steps + 1
