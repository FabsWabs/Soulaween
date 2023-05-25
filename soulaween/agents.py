import numpy as np
import os
import pathlib

from soulaween.utils.utils import compute_state_ind


class Agent():
    def __init__(self):
        pass

    def get_action(self):
        pass

    def _random_action(self, mask):
        legal_actions = np.argwhere(mask==1).flatten()
        return np.random.choice(legal_actions)
    
    def eval(self):
        pass

    def train(self):
        pass

class RandomAgent(Agent):
    def get_action(self, next_move, state, mask):
        return self._random_action(mask)


class RuleBased(Agent):
    def __init__(self):
        mat_path = os.path.join(pathlib.Path(__file__).parent, 'utils', 'complete_set.npz')
        self.complete_sets = np.load(mat_path)['a']
        
    def get_action(self, next_move, state, mask):
        state_ind = compute_state_ind(state)
        actions = np.argwhere(self.complete_sets[state_ind])
        if actions.size != 0:
            return actions[0][0]
        return self._random_action(mask)


class NetworkAgent(Agent):
    def __init__(self, models, random_action_prob=[0.,0.]):
        self.model = models
        for key in self.model.keys():
            self.model[key].eval()
        
        self.random_action_prob = {'place_stone': random_action_prob[0],
                                   'choose_set': random_action_prob[1]}

    def get_action(self, next_move, state, mask):
        p = np.random.rand()
        if p < self.random_action_prob[next_move]:
            return self._random_action(mask)
        output = self.model[next_move](state).detach().numpy()
        output = np.exp(output/10.0) * mask
        action_output = np.argmax(output)
        return action_output
    
    def get_action_nets(self):
        return self.model

    def set_action_nets(self, value_net):
        for key in value_net.keys():
            self.model[key].load_state_dict(value_net[key].state_dict())
        
    def set_random_prob(self, random_action_prob):
        self.random_action_prob = {'place_stone': random_action_prob[0],
                                   'choose_set': random_action_prob[1]}
    
    def eval(self):
        for key in self.model.keys():
            self.model[key].eval()
    
    def train(self):
        for key in self.model.keys():
            self.model[key].train()
    


