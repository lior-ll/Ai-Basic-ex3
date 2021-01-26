import random
from copy import deepcopy
import numpy as np
from itertools import combinations as comb
from itertools import product
from time import time
ids = ['111111111', '111111111']

DIMENSIONS = (10, 10)
class TikTok:
    def __init__(self):
        self.tik = None
    def set_tik(self):
        self.tik = time()
    def set_tok(self):
        print(time() - self.tik)
        self.tik = None
'''
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
'''
class Board:
    def __init__(self, initial_state ,control_zone, order):
        ''' state is paddaed zoc is not'''
        self.my_control_zone = control_zone
        self.annemy_control_zone = self.get_annemy_zoc(initial_state)
        self.order = ["first", "second"]
        if order == 'second':
            self.order.reverse()

    def pad_the_input(self, a_map):
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
    def simulate_annemy_turn(self, cur_state, new_map):
        action = self.extract_action_from_new_map(cur_state, new_map)
        #self.map_state = new_map
        return self.simulate_action_result(action, cur_state, player=1)#anammy 

    def extract_action_from_new_map(self, cur_state, new_map):
        action = []
        map_state = self.state_to_agent(cur_state)
        for i in  range(DIMENSIONS[0]):
            for j in range(DIMENSIONS[1]):
                if new_map[i][j] != map_state[i][j]:
                    if new_map[i][j] == "I":
                        action.append(('vaccinate', (i,j)))
                    if new_map[i][j] == "Q":
                        action.append(('quarantine', (i,j)))
        return action

    def get_annemy_zoc(self, board):
        #state = self.state_to_agent(cur_state)
        habitable_tiles = set([(i, j) for i, j in 
                   product(range(DIMENSIONS[0]),
                                             range(DIMENSIONS[1])) if 'U' not in board[i][j]])
        return habitable_tiles.symmetric_difference(self.my_control_zone)

    def state_to_agent(self, state):
        state_as_list = []
        for i in range(DIMENSIONS[0]):
            state_as_list.append([]*DIMENSIONS[1])
            for j in range(DIMENSIONS[1]):
                state_as_list[i].append(state[(i + 1, j + 1)][0])
        return state_as_list

    def apply_action(self, actions, state):
        '''action here is no padded'''
        if not actions:
            return state
        for atomic_action in actions:
            effect, location = atomic_action[0], (atomic_action[1][0] + 1, atomic_action[1][1] + 1)
            if 'v' in effect:
                state[location] = 'I'
            else:
                state[location] = 'Q0'
        return state

    def change_state(self, state):
        #new_state = deepcopy(self.state)
        new_state = state.copy()

        # virus spread
        for i in range(1, DIMENSIONS[0] + 1):
            for j in range(1, DIMENSIONS[1] + 1):
                if state[(i, j)] == 'H' and ('S' in state[(i - 1, j)] or
                                             'S' in state[(i + 1, j)] or
                                             'S' in state[(i, j - 1)] or
                                             'S' in state[(i, j + 1)]):
                    new_state[(i, j)] = 'S1'

        # advancing sick counters
        for i in range(1, DIMENSIONS[0] + 1):
            for j in range(1, DIMENSIONS[1] + 1):
                if 'S' in state[(i, j)]:
                    turn = int(state[(i, j)][1])
                    if turn < 3:
                        new_state[(i, j)] = 'S' + str(turn + 1)
                    else:
                        new_state[(i, j)] = 'H'

                # advancing quarantine counters
                if 'Q' in state[(i, j)]:
                    turn = int(state[(i, j)][1])
                    if turn < 2:
                        new_state[(i, j)] = 'Q' + str(turn + 1)
                    else:
                        new_state[(i, j)] = 'H'

        return new_state


    def game_over(self, state):
        sick_in_board = 'S1' in state.values() or 'S2' in state.values() or 'S3' in state.values()
        return not sick_in_board

    def get_all_actions(self, player, state):
        if player == 0:
            zoc = self.my_control_zone
        else:
            zoc = self.annemy_control_zone
        board = self.state_to_agent(state)
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
    
    def simulate_action_result(self, action, cur_state, player, force=False):
        state = self.apply_action(action, cur_state)
        if self.order[player] == "second" or force:
            state = self.change_state(state)
        return state

clk = TikTok()
class Agent:
    
    def __init__(self, initial_state, zone_of_control, order):
        print(initial_state)
        print(zone_of_control)
        print(order)
        self.first_time = order == 'first'
        self.board = Board(initial_state, zone_of_control, order)
        self.current_state = self.board.pad_the_input(initial_state)

    def act(self, a_map):
        state = self.board.pad_the_input(a_map)
        if self.first_time:
            self.first_time = False
        else:
            state = self.board.simulate_annemy_turn(self.current_state, a_map) # better to return board as well
        test_state = deepcopy(state)
        action = self.minimax(state, 1, float('-inf'), float('inf'), 0)[1]
        assert test_state == state
        state = self.board.simulate_action_result(action, state, 0) 
        self.current_state = state
        return action, self.board.state_to_agent(state)

    def get_all_game_options(self, state, player):
        #clk.set_tik()
        states = []
        for action in self.board.get_all_actions(player, state):
            new_state = state.copy()
            new_state = self.board.simulate_action_result(action, new_state, player)
            states.append((action, new_state))
        #print(f"player {player} number of options: {len(states)}")
        assert len(states) > 0
        
        #clk.set_tok()
        #if len(states) > 10:
            #print("cut actions")
            #return random.sample(states, 10)
        return states

    def evaluate(self, state, player):
        '''simpale eval'''
        board_map = self.board.state_to_agent(state)
        score = 0
        zones = [ (1, self.board.my_control_zone), (-1, self.board.annemy_control_zone) ]
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

    def minimax(self, state, depth, alpha, beta, player):
        if depth == 0 or self.board.game_over(state):
            if self.board.order[player] == 'second':
                state = self.board.change_state(state)
            return self.evaluate(state, player), []
        
        if player == 0:
            maxEval = float('-inf')
            best_action = []
            c = 0
            for action, res_state in self.get_all_game_options(state, player):
                c += 1
                evaluation = self.minimax(res_state, depth-1, alpha, beta, 1)[0]
                maxEval = max(maxEval, evaluation)
                if maxEval == evaluation:
                    best_action = action
                alpha = max(evaluation, alpha)
                if beta <= alpha:
                    #print(f"run just {c} times !!")
                    break
            return maxEval, best_action
        else:
            minEval = float('inf')
            best_action = []
            c = 0
            for action, res_state in self.get_all_game_options(state, player):
                c += 1
                evaluation = self.minimax(res_state, depth-1 ,alpha, beta, 0)[0]
                minEval = min(minEval, evaluation)
                if minEval == evaluation:
                    best_action = action
                beta = min(evaluation, beta)
                if beta <= alpha:
                    #print(f"run just {c} times !!")
                    break
            return minEval, best_action

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
