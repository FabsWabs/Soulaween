import time
import random
import numpy as np
import os

import torch

from soulaween.env.soulaween import Soulaween
from soulaween.agents import NetworkAgent, RandomAgent, RuleBased
from soulaween.utils.utils import get_networks

def read_action(legal_actions):
    legal_actions_str = [str(i) for i in legal_actions]
    action = input(f"Choose action from {legal_actions}: ")
    while True:
        if action not in legal_actions_str:
            print("Action not legal.")
            action = input(f"Choose action from {legal_actions}: ")
        else:
            break
    return int(action)
    

if __name__ == '__main__':
    env = Soulaween()

    players = ['Human', 'AI']

    agent_type = input("Choose Agent from [random, transformer, linear, rule_based]: ")
    while True:
        if agent_type not in ["random", "transformer", "linear", "rule_based"]:
            agent_type = input("Input invalid, choose from [random, transformer, linear, rule_based]: ")
        else:
            break
    
    if agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "transformer":
        models = -1
        agent = NetworkAgent(models)
    elif agent_type == "linear":
        load = "_5710_0.46.pt"
        load_model_folder = "20230518-200108"
        linear = True
        model_str = 'linear' if linear else 'transformer'
        load_model_folder = os.path.join("..", 'model_rl', model_str, load_model_folder)

        env = Soulaween()
        obs_space = env.get_obs_space()
        act_space = env.get_act_space()

        action_net, value_net = get_networks(linear, load, obs_space, act_space, load_model_folder)
        agent = NetworkAgent(action_net)
    elif agent_type == "rule_based":
        agent = RuleBased()

    state = env.reset()
    print(f"{players[env.current_player]} begins!")
    time.sleep(1)
    env.render()
    time.sleep(1)

    done = False

    while not done:
        state = torch.Tensor(state)
        cur_player = env.current_player

        if players[cur_player] == 'AI':
            next_move = env.next_move
            legal_actions = env.legal_actions
            action = agent.get_action(next_move, state, legal_actions)
            print(f"AI chose action {action}.")
            _ = input()
        else:
            legal_actions = [i for i in np.argwhere(env.legal_actions).flatten()]
            action = read_action(legal_actions)
            
        time.sleep(1)
        state, _, done, env_dict = env.step(action)
        env.render()
    print(f"The winner is the {players[env_dict['winner']]}!")
    print(f"Thanks for playing!")
    
