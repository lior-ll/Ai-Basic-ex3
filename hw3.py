import random
from copy import deepcopy
import numpy as np
from itertools import combinations as comb
from itertools import product

ids = ['111111111', '111111111']

DIMENSIONS = (10, 10)

def pad_the_input(a_map):
    state = {}
    new_i_dim = DIMENSIONS[0] + 2
    new_j_dim = DIMENSIONS[1] + 2
    for i in range(0, new_i_dim):
        for j in range(0, new_j_dim):
            if i == 0 or j == 0 or i == new_i_dim - 1 or j == new_j_dim - 1:
                state[(i, j)] = 'U'
            elif 'S' in a_map[i - 1][j - 1]:
                state[(i, j)] = 'S1'
            else:
                state[(i, j)] = a_map[i - 1][j - 1]
    return state

class Board:
    def __init__(self, a_map, control_zone, order):
        ''' state is paddaed zoc is not'''
        #self.map_state = a_map
        self.state = pad_the_input(a_map) 
        #self.state = deepcopy(self.initial_state)
        # parse from zone of control
        self.my_control_zone = control_zone
        self.annemy_control_zone = self.get_annemy_zoc()
        self.order = ["first", "second"]
        if order == 'second':
            self.order.reverse()

    def simulate_annemy_turn(self, new_map):
        action = self.extract_action_from_new_map(new_map)
        #self.map_state = new_map
        self.simulate_action_result(action, player=1)#anammy 

    def extract_action_from_new_map(self, new_map):
        action = []
        map_state = self.state_to_agent()
        for i in  range(DIMENSIONS[0]):
            for j in range(DIMENSIONS[1]):
                if new_map[i][j] != map_state[i][j]:
                    if new_map[i][j] == "I":
                        action.append(('vaccinate', (i,j)))
                    if new_map[i][j] == "Q":
                        action.append(('quarantine', (i,j)))
        return action
    def get_annemy_zoc(self):
        state = self.state_to_agent()
        habitable_tiles = set([(i, j) for i, j in 
                   product(range(DIMENSIONS[0]),
                                             range(DIMENSIONS[1])) if 'U' not in state[i][j]])
        return habitable_tiles.symmetric_difference(self.my_control_zone)
    def state_to_agent(self):
        state_as_list = []
        for i in range(DIMENSIONS[0]):
            state_as_list.append([]*DIMENSIONS[1])
            for j in range(DIMENSIONS[1]):
                state_as_list[i].append(self.state[(i + 1, j + 1)][0])
        return state_as_list

    def apply_action(self, actions):
        '''action here is no padded'''
        if not actions:
            return
        for atomic_action in actions:
            effect, location = atomic_action[0], (atomic_action[1][0] + 1, atomic_action[1][1] + 1)
            if 'v' in effect:
                self.state[location] = 'I'
            else:
                self.state[location] = 'Q0'

    def change_state(self):
        new_state = deepcopy(self.state)

        # virus spread
        for i in range(1, DIMENSIONS[0] + 1):
            for j in range(1, DIMENSIONS[1] + 1):
                if self.state[(i, j)] == 'H' and ('S' in self.state[(i - 1, j)] or
                                                  'S' in self.state[(i + 1, j)] or
                                                  'S' in self.state[(i, j - 1)] or
                                                  'S' in self.state[(i, j + 1)]):
                    new_state[(i, j)] = 'S1'

        # advancing sick counters
        for i in range(1, DIMENSIONS[0] + 1):
            for j in range(1, DIMENSIONS[1] + 1):
                if 'S' in self.state[(i, j)]:
                    turn = int(self.state[(i, j)][1])
                    if turn < 3:
                        new_state[(i, j)] = 'S' + str(turn + 1)
                    else:
                        new_state[(i, j)] = 'H'

                # advancing quarantine counters
                if 'Q' in self.state[(i, j)]:
                    turn = int(self.state[(i, j)][1])
                    if turn < 2:
                        new_state[(i, j)] = 'Q' + str(turn + 1)
                    else:
                        new_state[(i, j)] = 'H'

        self.state = new_state

    def update_scores(self, player, control_zone):
        for (i, j) in control_zone:
            if 'H' in self.state[(i, j)]:
                self.score[player] += 1
            if 'I' in self.state[(i, j)]:
                self.score[player] += 1
            if 'S' in self.state[(i, j)]:
                self.score[player] -= 1
            if 'Q' in self.state[(i, j)]:
                self.score[player] -= 5

    def game_over(self):
        sick_in_board = 'S1' in self.state.values() or 'S2' in self.state.values() or 'S3' in self.state.values()
        return not sick_in_board

    def get_all_actions(self, player):
        if player == 0:
            zoc = self.my_control_zone
        else:
            zoc = self.annemy_control_zone
        board = self.state_to_agent()
        #bulid medics options
        board = np.array(board)
        row, col = np.where(board == 'H')
        medics_options = set(zip(row, col))
        medics_options.intersection_update(zoc)
        info = ["vaccinate"]*len(medics_options)
        medics_options = list(zip(info, medics_options)) #what if empty #cordinate in board
        comb_factor = min(1, len(medics_options))
        medics_options = list(comb(medics_options, comb_factor)) # take into acount number of mdics teams (nCk)

        #bulid police options
        row, col = np.where(board == 'S')
        police_options =set(zip(row, col))
        police_options.intersection_update(zoc)
        info = ["quarantine"]*len(police_options)
        police_options =list(zip(info, police_options))
        comb_factor = min(2, len(police_options))
        police_options = list(comb(police_options, comb_factor)) # here we may want to chose only one or empty - consider to add this

        #build actions sets
        actions = list(product(medics_options, police_options)) #medics_options X police_options (cartesian product)
        tuple_united = lambda x : x[0] + x[1]
        actions = [tuple_united(action) for action in actions]
        return actions
    
    def simulate_action_result(self, action, player):
        self.apply_action(action)
        if self.order[player] == "second":
            self.change_state()

'''    
def game_over(board):
    board = np.array(board)
    sick_in_board = (board == 'S').any()
    return not sick_in_board

def get_all_actions(board, zoc):
        #bulid medics options
        board = np.array(board)
        row, col = np.where(board == 'H')
        medics_options = set(zip(row, col))
        medics_options.intersection_update(zoc)
        info = ["vaccinate"]*len(medics_options)
        medics_options = list(zip(info, medics_options)) #what if empty #cordinate in board
        comb_factor = min(1, len(medics_options))
        medics_options = list(comb(medics_options, comb_factor)) # take into acount number of mdics teams (nCk)

        #bulid police options
        row, col = np.where(board == 'S')
        police_options =set(zip(row, col))
        police_options.intersection_update(zoc)
        info = ["quarantine"]*len(police_options)
        police_options =list(zip(info, police_options))


        comb_factor = min(2, len(police_options))
        police_options = list(comb(police_options, comb_factor)) # here we may want to chose only one or empty - consider to add this

        #build actions sets
        actions = list(product(medics_options, police_options)) #medics_options X police_options (cartesian product)
        tuple_united = lambda x : x[0] + x[1]
        actions = [tuple_united(action) for action in actions]
        return actions
'''
def get_all_game_options(board, player):
    boards = []
    for action in board.get_all_actions(player):
        new_board = deepcopy(board)
        new_board.simulate_action_result(action, player)
        boards.append((action, new_board))
    print(f"player {player} number of options: {len(boards)}")
    assert len(boards) > 0
    return boards

def evaluate(board, player):
    '''simpale eval'''
    board_map = board.state_to_agent()
    score = 0
    zones = [ (1, board.my_control_zone), (-1, board.annemy_control_zone) ]
    for f, control_zone in zones:
        for (i, j) in control_zone:
            if 'H' == board_map[i][j]:
                score += f*1
            if 'I' == board_map[i][j]:
                score += f*1
            if 'S' == board_map[i][j]:
                score -= f*1
            if 'Q' == board_map[i][j]:
                score -= f*5
        #print(f"eval: player {player}, score {score}")
    return score

def minimax(board, depth, player):
    if depth == 0 or board.game_over():
        return evaluate(board, player), []
    
    if player == 0:
        maxEval = float('-inf')
        best_action = []
        for action, res_board in get_all_game_options(board, player):
            evaluation = minimax(res_board, depth-1, 1)[0]
            maxEval = max(maxEval, evaluation)
            if maxEval == evaluation:
                best_action = action
        
        return maxEval, best_action
    else:
        minEval = float('inf')
        best_action = []
        for action, res_board in get_all_game_options(board, player):
            evaluation = minimax(res_board, depth-1, 0)[0]
            minEval = min(minEval, evaluation)
            if minEval == evaluation:
                best_action = action
        
        return minEval, best_action

class Agent:
    def __init__(self, initial_state, zone_of_control, order):
        print(initial_state)
        print(zone_of_control)
        print(order)
        self.first_time = order == 'first'
        self.board = Board(initial_state, zone_of_control, order)
        
    def act(self, state):
        if self.first_time:
            self.first_time = False
        else:
            self.board.simulate_annemy_turn(state) # better to return board as well
        action = minimax(self.board, 1, 0)[1]
        self.board.simulate_action_result(action, 0)
        #for line in self.board.state_to_agent():
        #    print(line)
        return action, self.board.state_to_agent()


# implementation of a random agent
class AgentR:
    def __init__(self, initial_state, zone_of_control, order):
        self.zoc = zone_of_control
        print(initial_state)

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
