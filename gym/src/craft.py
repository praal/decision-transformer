import logging
from random import Random
from typing import FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .rl.environment import Environment, Observation, State
import numpy as np


ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # down
    (0, -1),  # up
    (-1, 0),  # left
    (1, 0),   # right
    (0, 0),   # do
]

OBJECTS = dict([(v, k) for k, v in enumerate(
   ["wood", "iron", "gold", "gem"])])

def update_facts(facts: Sequence[bool], objects: Observation, graph, do_action = False) -> Set[int]:
    state = set([i for i, v in enumerate(facts) if v])
    if not do_action:
        return state
    for o in objects:
        if o in OBJECTS:
            ind = OBJECTS[o]
            to_add = True
            for i in range(len(graph)):
                if graph[i][ind] == 1 and i not in state:
                    to_add = False
            if to_add:
                state.add(OBJECTS[o])

    return state


class CraftState(State):
    facts: Tuple[bool, ...]
    map_data: Tuple[Tuple[Observation, ...], ...]

    def __init__(self, x: int, y: int, facts: Set[int]):
        self.x = x
        self.y = y

        fact_list = [False] * len(OBJECTS)
        for fact in facts:
            fact_list[fact] = True
        self.facts = tuple(fact_list)
        self.uid = (self.x, self.y, self.facts)

    def __str__(self) -> str:
        return "({:2d}, {:2d}, {})".format(self.x, self.y, self.facts)

    @staticmethod
    def random(rng: Random,
               map_data: Sequence[Sequence[Observation]]) -> 'CraftState':
        # return CraftState(5, 5, set())
        while True:
            y = 1
            x = 1
            if "wall" not in map_data[y][x]:
                return CraftState(x, y,())


MAPPING: Mapping[str, FrozenSet[str]] = {
    'A': frozenset(),
    'X': frozenset(["wall"]),
    'w': frozenset(["wood"]),
    'f': frozenset(["iron"]),
    'g': frozenset(["gold"]),
    'h': frozenset(["gem"]),
    ' ': frozenset(),
    }


def load_map(map_fn: str) -> Tuple[Tuple[Observation, ...], ...]:
    with open(map_fn) as map_file:

        array = []
        for l in map_file:
            if len(l.rstrip()) == 0:
                continue

            row = []
            for cell in l.rstrip():
                row.append(MAPPING[cell])
            array.append(tuple(row))

    return tuple(array)


class Craft(Environment):
    map_data: Tuple[Tuple[Observation, ...], ...]
    num_actions = 5

    def __init__(self, map_fn: str, rng: Random, graph = None, order = None, causal=False):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        self.graph = graph
        self.causal = causal
        self.order = order
        if self.graph is None:
            self.order = np.random.permutation(len(OBJECTS))
            self.graph = np.zeros([len(OBJECTS), len(OBJECTS)])
            for i in range(len(self.order) - 1):
                self.graph[self.order[i]][self.order[i+1]] = 1

        super().__init__(CraftState.random(self.rng, self.map_data))

    def step(self, a):
        x = self.state.x + ACTIONS[a][0]
        y = self.state.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= self.width or y >= self.height or \
                "wall" in self.map_data[y][x]:
            reward, done = self.cost(self.state, a, self.state)
            ret_state = self.get_one_hot_state()
            return ret_state, reward, done, ""

        objects = self.map_data[y][x]
        new_facts = update_facts(self.state.facts, objects, self.graph, (a == self.num_actions - 1))
        reward, done = self.cost(self.state, a, CraftState(x, y, new_facts))
        self.state = CraftState(x, y, new_facts)
        logging.debug("success, current state is %s", self.state)
        ret_state = self.get_one_hot_state()
        return ret_state, reward, done, ""

    def cost(self, s0: CraftState, a: int, s1: CraftState):
        cnt0 = 0
        cnt1 = 0
        all_done = True
        cost = 0
        for fact in s0.facts:
            if fact is True:
                cnt0 += 1
        for fact in s1.facts:
            if fact is True:
                cnt1 += 1
            else:
                all_done = False
        if cnt1 > cnt0:
            cost = 1
        if all_done:
            cost = 1

        return cost, all_done

    def observe(self, state: CraftState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def get_one_hot_state(self):
        mat = np.zeros((self.height - 2, self.width - 2))
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if "wood" in self.map_data[y][x]:
                    mat[y - 1][x - 1] = 2
                if "iron" in self.map_data[y][x]:
                    mat[y - 1][x - 1] = 3
                if "gold" in self.map_data[y][x]:
                    mat[y - 1][x - 1] = 4
                if "gem" in self.map_data[y][x]:
                    mat[y - 1][x - 1] = 5
        mat[self.state.y - 1][self.state.x - 1] = 1
        masking = len(OBJECTS) + 2
        flat_mat = mat.reshape(-1).copy().astype(int)
        one_hot = np.zeros((flat_mat.size, masking))
        one_hot[np.arange(flat_mat.size), flat_mat] = 1
        one_hot = one_hot.reshape(-1).copy()
        causal = np.zeros((1, masking))
        causal = causal.reshape(-1).copy()
        if self.causal:
            true_facts = 0
            next_goal = 0
            for f in self.state.facts:
                if f:
                    true_facts += 1

            if true_facts < len(OBJECTS):
                next_goal = self.order[true_facts] + 2
            goal_one_hot = np.zeros((1, masking))
            goal_one_hot[0][next_goal] = 1
            goal_one_hot = goal_one_hot.reshape(-1).copy()
            #one_hot = np.concatenate([one_hot, goal_one_hot])
            causal = goal_one_hot
        return [one_hot, causal]

    def observation_space(self):
        return self.get_one_hot_state()

    def action_space(self):
        return self.num_actions

    def reset(self, state: Optional[CraftState] = None, graph = None):
        if state is not None:
            self.state = state
            #if self.graph is None:
            self.order = np.random.permutation(len(OBJECTS))
            self.graph = np.zeros([len(OBJECTS), len(OBJECTS)])
            for i in range(len(self.order) - 1):
                self.graph[self.order[i]][self.order[i + 1]] = 1
        elif graph is not None:
            self.state = CraftState.random(self.rng, self.map_data)
            self.graph = graph
        else:
            self.state = CraftState.random(self.rng, self.map_data)
            #if self.graph is None:
            self.order = np.random.permutation(len(OBJECTS))
            self.graph = np.zeros([len(OBJECTS), len(OBJECTS)])
            for i in range(len(self.order) - 1):
                self.graph[self.order[i]][self.order[i+1]] = 1
                
        return self.get_one_hot_state()


