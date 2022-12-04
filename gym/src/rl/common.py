from typing import Iterable, Tuple, Union

from .environment import RewardFn

class ReachFacts(RewardFn):
    target: Tuple[int, ...]

    def __init__(self, environment, facts: Iterable[int]):
        super().__init__(environment)
        self.target = tuple(facts)

    def __call__(self, s0, a,s1) -> Tuple[float, bool]:
        cost,_ = self.environment.cost(s0, a, s1)
        for fact in self.target:
            if not s1.facts[fact]:
                return cost, False
        return cost, True

    def reset(self):
        pass
