import numpy as np

class RandomAgent():
    def __init__(self):
        pass

    def get_action(self, next_move, state, mask):
        legal_actions = np.argwhere(mask==1).flatten()
        return np.random.choice(legal_actions)


class TransformerAgent():
    def __init__(self, models, random_action_prob=[0.,0.]):
        self.model = models
        for key in self.model.keys():
            self.model[key].eval()
        
        self.random_action_prob = {'place_stone':random_action_prob[0],
                                   'choose_set':random_action_prob[1]}

    def get_action(self, next_move, state, mask):
        p = np.random.rand()
        if p < self.random_action_prob[next_move]:
            return self.__random_action(mask)
        output = self.model[next_move](state).detach().numpy()
        output = np.exp(output/10.0) * mask
        action_output = np.argmax(output)
        return action_output
    
    def __random_action(self, mask):
        legal_actions = np.argwhere(mask==1).flatten()
        return np.random.choice(legal_actions)
    
    def set_action_nets(self, value_net):
        for key in value_net.keys():
            self.model[key].load_state_dict(value_net[key].state_dict())
        
    def set_random_prob(self, random_action_prob):
        self.random_action_prob = {'place_stone':random_action_prob[0],
                                   'choose_set':random_action_prob[1]}

