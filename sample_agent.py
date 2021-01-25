import random

ids = ['AI']


class Agentsi:
    def __init__(self, initial_state, zone_of_control, order):
        self.zoc = zone_of_control

    def act(self, state):
        action = []
        healthy = set()
        sick = set()
        for (i, j) in self.zoc:
            if 'H' in state[i][j]:
                healthy.add((i, j))
            if 'S' in state[i][j]:
                sick.add((i, j))
        try:
            to_quarantine = random.sample(sick, 2)
        except ValueError:
            to_quarantine = []
        try:
            to_vaccinate = random.sample(healthy, 1)
        except ValueError:
            to_vaccinate = []
        for item in to_quarantine:
            action.append(('quarantine', item))
        for item in to_vaccinate:
            action.append(('vaccinate', item))

        return action

import random

ids = ['AI1']


class Agent:
    def __init__(self, initial_state, zone_of_control, order):
        self.zoc = zone_of_control

    def process_state(self, state):
        healthy = []
        sick = []
        for (i, j) in self.zoc:
            if 'H' in state[i][j]:
                healthy.append((i, j))
            if 'S' in state[i][j]:
                sick.append((i, j))
        return healthy, sick

    def sick_heuristic(self, healthy, coordinate):
        neighbors = ((coordinate[0] - 1, coordinate[1]),
                     (coordinate[0] + 1, coordinate[1]),
                     (coordinate[0], coordinate[1] - 1),
                     (coordinate[0], coordinate[1] + 1))
        h = sum(1 for neighbor in neighbors if (neighbor in healthy and neighbor in self.zoc))
        return h

    def act(self, state):
        action = []
        healthy, sick = self.process_state(state)
        sick.sort(key=lambda x: self.sick_heuristic(healthy, x), reverse=True)
        try:
            to_quarantine = (sick[:2])
        except KeyError:
            to_quarantine = []
        try:
            to_vaccinate = random.sample(healthy, 1)
        except ValueError:
            to_vaccinate = []

        for item in to_quarantine:
            action.append(('quarantine', item))
        for item in to_vaccinate:
            action.append(('vaccinate', item))
        return action

