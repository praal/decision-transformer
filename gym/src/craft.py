import logging
from random import Random
from typing import FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .environment import Environment, Observation, State
import numpy as np

ACTIONS: List[Tuple[int, int]] = [
    (0, 1),   # down
    (0, -1),  # up
    (-1, 0),  # left
    (1, 0),   # right
    (0, 0),   # do
]

OBJECTS = dict([(v, k) for k, v in enumerate(
   ["wood", "iron", "gold"])])



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

    def __init__(self, map_fn: str, rng: Random, graph = None):
        self.map_data = load_map(map_fn)
        self.height = len(self.map_data)
        self.width = len(self.map_data[0])
        self.rng = rng
        self.graph = graph
        if graph is None:
            #order = np.random.permutation(len(OBJECTS))
            order = [0, 1, 2]
            self.graph = np.zeros([len(OBJECTS), len(OBJECTS)])
            for i in range(len(order) - 1):
                self.graph[order[i]][order[i+1]] = 1

        super().__init__(CraftState.random(self.rng, self.map_data))

    def step(self, a):
        x = self.state.x + ACTIONS[a][0]
        y = self.state.y + ACTIONS[a][1]
        logging.debug("applying action %s:%s", a, ACTIONS[a])
        if x < 0 or y < 0 or x >= self.width or y >= self.height or \
                "wall" in self.map_data[y][x]:
            reward, done = self.cost(self.state, a, self.state)
            ret_state = np.array([self.state.uid[0], self.state.uid[1]] + [int(elem) for elem in self.state.uid[2]])
            return ret_state, reward, done, ""

        objects = self.map_data[y][x]
        new_facts = update_facts(self.state.facts, objects, self.graph, (a == self.num_actions - 1))
        reward, done = self.cost(self.state, a, CraftState(x, y, new_facts))
        self.state = CraftState(x, y, new_facts)
        logging.debug("success, current state is %s", self.state)
        ret_state = np.array([self.state.uid[0], self.state.uid[1]] + [int(elem) for elem in self.state.uid[2]])
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

        #if all_done:
          #  if s1.x == self.width - 2 and s1.y == self.height - 2:
          #      all_done = True
          #  else:
  
        if all_done:
            cost = 1

        return cost, all_done

    def observe(self, state: CraftState) -> Observation:
        return self.map_data[self.state.y][self.state.x]

    def reset(self, state: Optional[CraftState] = None, graph = None):
        if state is not None:
            self.state = state
            self.graph = graph
            if self.graph is None:
                #order = np.random.permutation(len(OBJECTS))
                order = [0, 1, 2]
                self.graph = np.zeros([len(OBJECTS), len(OBJECTS)])
                for i in range(len(order) - 1):
                    self.graph[order[i]][order[i + 1]] = 1
        else:
            self.state = CraftState.random(self.rng, self.map_data)
            #order = np.random.permutation(len(OBJECTS))
            order = [0, 1, 2]
            self.graph = np.zeros([len(OBJECTS), len(OBJECTS)])
            for i in range(len(order) - 1):
                self.graph[order[i]][order[i+1]] = 1
        return np.array([self.state.uid[0], self.state.uid[1]] + [int(elem) for elem in self.state.uid[2]])

    @staticmethod
    def label(state: CraftState) -> FrozenSet[int]:
        return frozenset([i for i in range(len(OBJECTS)) if state.facts[i]])
