import numpy as np

from soulaween.utils import *
from soulaween.agents import RandomAgent, NetworkAgent
from soulaween.networks.transformer_networks import *
from soulaween.env.soulaween import Soulaween

def test_random_agent():
    agent = RandomAgent(np.array(32))
    trace_simulator = TraceSimulator(agent)
    sample_dict = trace_simulator.random_make_games(1)
    print("Tested Random Agent on 1 game.")

def test_transformer_agent():
    env = Soulaween()
    action_net = {
        'place_stone': PlaceStoneTransformer(env.get_act_space()['place_stone']),
        'choose_set': ChooseSetTransformer(env.get_act_space()['choose_set'])
    }
    agent = NetworkAgent(action_net)
    trace_simulator = TraceSimulator(agent, env)
    sample_dict = trace_simulator.random_make_games(1)
    obs = sample_dict['choose_set'][0].state
    print("Tested Transformer Agent on 1 game.")

if __name__ == "__main__":
    test_random_agent()
    test_transformer_agent()
