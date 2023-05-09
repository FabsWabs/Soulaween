import time
import random
import numpy as np

import torch

from soulaween.env.soulaween import Soulaween
from soulaween.agents import NetworkAgent, RandomAgent

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

    agent_type = input("Choose Agent from [random, transformer]: ")
    while True:
        if agent_type not in ["random", "transformer"]:
            agent_type = input("Input invalid, choose from [random, transformer]: ")
        else:
            break
    
    if agent_type == "random":
        agent = RandomAgent()
    elif agent_type == "transformer":
        models = -1
        agent == NetworkAgent(models)

    agent_player_number = 1
    
    state = env.reset()
    print(f"Player {env.current_player} begins!")
    time.sleep(1)
    env.render()
    time.sleep(1)

    done = False

    while not done:
        state = torch.Tensor(state)
        player = env.current_player

        if agent_player_number == player:
            next_move = env.next_move
            legal_actions = env.legal_actions
            action = agent.get_action(next_move, state, legal_actions)
            print(f"Agent chose action {action}.")
            _ = input()
        else:
            legal_actions = [i for i in np.argwhere(env.legal_actions).flatten()]
            action = read_action(legal_actions)
            
        time.sleep(1)
        state, _, done, env_dict = env.step(action)
        env.render()
    print(f"The winner is Player {env_dict['winner']}!")
    print(f"Thanks for playing!")
    
