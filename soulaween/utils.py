import time
import os
import random
from collections import namedtuple
import numpy as np
from datetime import datetime

import torch

from soulaween.env.soulaween import Soulaween

def time_str():
    return time.strftime("%Y%m%d-%H%M%S",time.localtime())


def print_log(log_str, log_path):
    with open(log_path, 'a') as f:
        print(log_str)
        print(log_str, file=f)

TraceSlot = namedtuple(
    'TraceSlot', ['key','state','action'])

Transition = namedtuple(
    'Transition', ['state','action','reward'])

class Buffer():
    def __init__(self):
        self.memory = {'place_stone':[], 'choose_set':[]}
    
    def extend(self, data_dict):
        for key in data_dict.keys():
            self.memory[key].extend(data_dict[key])
    
    def get_batch(self, key, batch_size):
        n_batch = len(self.memory[key]) // batch_size
        ind_list = [ii for ii in range(n_batch)]
        random.shuffle(ind_list)
        for ii in ind_list:
            yield self.memory[key][ii:n_batch * batch_size:n_batch]
    
    def clear(self):
        self.__init__()


class TraceSimulator():
    def __init__(self, agent, env=Soulaween(),
                 record_state=['place_stone', 'choose_set'], 
                 discount=1):
        self.agent = {0:agent, 1:agent}
        self.env = env
        
        self.record_state = record_state
        self.discount = discount

        self.buffer = {'place_stone':[], 'choose_set':[]}
    
    def random_make_games(self, n_games):
        self.buffer = {'place_stone':[], 'choose_set':[]}
        for i in range(n_games):
            self.make_game_trace()
            print(f"Made Game {i}.")

        return self.buffer
    
    def make_game_trace(self):
        state = self.env.reset()
        done = False
        
        trace = {0:[], 1:[]}
        while not done:
            state = torch.Tensor(state)
            player = self.env.current_player
            next_move = self.env.next_move
            legal_actions = self.env.legal_actions
            action = self.agent[player].get_action(next_move, state, legal_actions)
            if player in [0, 1] and (next_move in self.record_state) and (action is not None):
                trace[player].append(TraceSlot(
                    key = next_move,
                    state = state.clone(), 
                    action = action))
            state, _, done, _ = self.env.step(action)
        scaled_point_dif = (10 + self.env.sets[0] - self.env.sets[1]) / 20
        reward = {0: scaled_point_dif, 1: 10 - scaled_point_dif}
            
        for player in [0, 1]:
            for rev_step in range(len(trace[player])):
                key = trace[player][-rev_step-1].key
                action = trace[player][-rev_step-1].action
                self.buffer[key].append(Transition(
                    state = trace[player][-rev_step-1].state.clone(), 
                    action = action, 
                    reward = reward[player] * (self.discount ** rev_step)))
        return

class Arena():
    def __init__(self, agent_1, agent_2):
        self.agent = {0: agent_1, 1: agent_2}
        self.env = Soulaween()
        
        self.match_points = [[], []]
        self.winner = []
    
    def multi_game_test(self, n_games, clear_result=True):
        if clear_result:
            self.reset()
        for _ in range(n_games):
            self.__duel()
        self.n_wins = [np.sum(np.array(self.winner) == player) for player in [0, 1]]
        self.win_rates = [n / np.sum(self.n_wins) for n in self.n_wins]
        return self.n_wins, self.match_points
        
    def __duel(self):
        state = self.env.reset()
        done = False
        
        while not done:
            state = torch.Tensor(state)
            player = self.env.current_player
            next_move = self.env.next_move
            legal_actions = self.env.legal_actions
            action = self.agent[player].get_action(next_move, state, legal_actions)
            state, _, done, _ = self.env.step(action)

        self.match_points[0].append(self.env.sets[0])
        self.match_points[1].append(self.env.sets[1])
        self.winner.append(np.argmax(self.env.sets))
    
    def test_result_str(self):
        assert len(self.winner) > 0
        win_num = self.n_wins
        win_rate = self.win_rates
        s = f'{np.sum(win_num)} games tested: '
        s += f'{win_num[0]} wins, {win_num[1]} losses; '
        s += f'Win rate: {win_rate[0] * 100:.2f} % | '
        s += f'{np.mean(self.match_points[0]):.1f} points on average'
        return s
                    
    def reset(self):
        self.match_points = [[], []]
        self.winner = []
        return 

def render_board_from_obs(obs):
    board = np.zeros(16)
    board[np.argwhere(obs[1]==1)] = 1
    board[np.argwhere(obs[2]==1)] = -1
    next_move = 'place_stone' if obs[3,0] == 1 else 'choose_set'

    print('-------------------------------------------------')
    board_str = '┌───┬───┬───┬───┐\n'
    for i in range(16):
        if board[i] == 1:
            board_str += '│ X '
        elif board[i] == -1:
            board_str += '│ O '
        else:
            board_str += '│ . '
        

        if i % 4 == 3 and i < 14:
            if i == 3:
                board_str += f'│\n'
            elif i == 7:
                board_str += f'│\n'
            else:
                board_str += f'│\n'
            board_str += '├───┼───┼───┼───┤\n'
    board_str += '│\n└───┴───┴───┴───┘'
    print(board_str)
    print(f'Next Move: {next_move}\n')


def parallel_sampling(agent, n_games):
    trace_simulator = TraceSimulator(agent)
    sample_dict = trace_simulator.random_make_games(n_games)
    return sample_dict

def parallel_arena_test(agent, opponent, n_games):
    arena = Arena(agent, opponent)
    n_wins, points = arena.multi_game_test(n_games)
    result = n_wins
    point_dif = np.array(points[0]) - np.array(points[1])
    result.append(np.mean(point_dif))
    return result

def arena_analysis(result, log_path=None):
    result = np.array(result)
    win_num = np.sum(result[:,:2], axis=0)
    win_rate = win_num / np.sum(win_num)
    mean_point_dif = np.mean(result[:,2])
    if log_path:
        header = f'{np.sum(win_num, dtype=np.int8)} games tested.'
        print_log(header, log_path)
        s = f'   {int(win_num[0])} wins, {int(win_num[1])} losses; '
        s += f'Win rate: {win_rate[0] * 100:.2f} % | '
        s += f'Mean point difference: {mean_point_dif:.1f} '
        print_log(s, log_path)
    return mean_point_dif


def random_action_prob_scheduler(score):
    if score < -3:
        p = [0.20] * 2
    elif score < -1:
        p = [0.15] * 2
    elif score < 2:
        p = [0.125] * 2
    elif score < 5:
        p = [0.10] * 2
    elif score < 8:
        p = [0.075] * 2
    else:
        p = [0.05] * 2
    return p