import numpy as np
    

class Soulaween():
    def __init__(self):
        self.grid_length = 4
        self.win_condition = 10
        self.num_squares = self.grid_length * self.grid_length
        self.act_space = {'place_stone': np.array(32),
                          'choose_set': np.array(10)}
        self.obs_space = np.array([7, 16])
        self.possible_sets = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            [0, 5, 10, 15],
            [12, 9, 6, 3]
        ])

    @property
    def observation(self):
        empty = np.where(self.board==0, 1, 0)
        cross = np.where(self.board==1, 1, 0)
        circle = np.where(self.board==-1, 1, 0)
        place_stone = np.ones((self.num_squares,)) * np.where(self.next_move =='place_stone', 1, 0)
        choose_set = np.ones((self.num_squares,)) * np.where(self.next_move =='choose_set', 1, 0)
        my_points = np.ones((self.num_squares,)) * self.sets[self.current_player_num]
        enemy_points = np.ones((self.num_squares,)) * self.sets[(self.current_player_num + 1) % 2]
        start_player = np.ones((self.num_squares,)) \
            if self.current_player_num == 0 else np.zeros((self.num_squares,))
        sweeps = np.ones((self.num_squares,)) * self.sweeps
        f = np.vstack((empty, cross, circle, place_stone, choose_set, my_points, enemy_points, \
            start_player, sweeps))
        return f

    @property
    def legal_actions(self):
        if self.next_move == 'place_stone':
            free_squares = np.argwhere(self.board==0)
            indices = np.concatenate((free_squares, free_squares + 16)).flatten()
            out = np.zeros((32,))
            out[indices] = 1
        else:
            indices = np.fromiter(self._check_sets().keys(), dtype=np.int8)
            out = np.zeros((10,))
            out[indices] = 1
        return out 

    @property
    def current_player(self):
        return self.current_player_num
    
    @property
    def game_status(self):
        return self.sets

    def _check_sets(self):
        sets = dict()
        for i, candidate in enumerate(self.possible_sets):
            if np.all(self.board[candidate]==1) or np.all(self.board[candidate]==-1):
                sets[i] = candidate
        return sets
    
    def _board_full(self):
        return not np.any(self.board == 0)
    
    def _clean_board(self, action):
        remove = 1 if action < 16 else -1
        self.board = np.where(self.board==remove, 0, self.board)
    
    def get_act_space(self):
        return self.act_space

    def step(self, action):
        reward = 0
        if self.next_move == 'place_stone':
            square = action % self.num_squares
            self.board[square] = 1 if action < 16 else -1

            flip = []
            if square not in [0, 4, 8, 12]:
                flip.append(square - 1)
            if square not in [0, 1, 2, 3]:
                flip.append(square - 4)
            if square not in [3, 7, 11, 15]:
                flip.append(square + 1)
            if square not in [12, 13, 14, 15]:
                flip.append(square + 4)
            
            for flip_square in flip:
                self.board[flip_square] *= -1
            
            sets = self._check_sets()
            if len(sets) < 2:
                if len(sets) == 1:
                    self.sets[self.current_player_num] += 1
                    self.board[list(sets.values())[0]] = 0
                if self._board_full():
                    self._clean_board(action)
                    self.sweeps += 1
                    if self.sweeps == 10:
                        self.done = True
                        return self.observation, reward, self.done, {}
                self.current_player_num = 1 if self.current_player_num == 0 else 0
                self.turns_taken += 1
            else:
                self.next_move = 'choose_set'
                return self.observation, reward, self.done, {}
        else:   # self.next_move == 'choose_set':
            self.sets[self.current_player_num] += 1
            sets = self._check_sets()
            self.board[sets[action]] = 0
            self.current_player_num = 1 if self.current_player_num == 0 else 0
            self.turns_taken += 1
            self.next_move = 'place_stone'
        if np.any(self.sets >= self.win_condition):
            self.done = True
        reward = self.sets[self.current_player_num]
        return self.observation, reward, self.done, {}


    def reset(self):
        self.board = np.zeros(self.num_squares, dtype=np.int8)
        self.turns_taken = 0
        self.done = False
        self.current_player_num = 0
        self.sets = np.array([0, 0])
        self.next_move = 'place_stone'
        self.sweeps = 0
        return self.observation

    def render(self, mode='human'):
        print('-------------------------------------------------')
        board_str = '┌───┬───┬───┬───┐\n'
        for i in range(self.num_squares):
            if self.board[i] == 1:
                board_str += '│ X '
            elif self.board[i] == -1:
                board_str += '│ O '
            else:
                board_str += '│ . '
            

            if i % 4 == 3 and i < 14:
                if i == 3:
                    board_str += f'│       Achieved Sets: {self.sets}\n'
                elif i == 7:
                    board_str += f'│       Played Moves:  {self.turns_taken}\n'
                else:
                    board_str += f'│       Sweeps:        {self.sweeps}\n'
                board_str += '├───┼───┼───┼───┤\n'
        board_str += '│\n└───┴───┴───┴───┘'
        print(board_str)
        print(f'    → Player {self.current_player_num + 1}:   {self.next_move}\n')

