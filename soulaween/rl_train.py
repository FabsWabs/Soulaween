import torch
import numpy as np
import random
from collections import namedtuple

import os
import time
import pickle
import multiprocessing

from soulaween.env.soulaween import Soulaween
from soulaween.utils import Buffer, print_log, time_str, parallel_sampling, Transition, random_action_prob_scheduler, parallel_arena_test, arena_analysis
from soulaween.networks import PlaceStoneNet, ChooseSetNet, TargetQNet
from soulaween.agents import TransformerAgent, RandomAgent

if __name__ == '__main__':
    rl_folder = 'model_rl'
    if not os.path.isdir(rl_folder):
        os.mkdir(rl_folder)
    log_folder = 'logs'
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    log_path = os.path.join(log_folder, f'{time_str()}_log_rl.txt')
    saved_model_path = {'place_stone': os.path.join(rl_folder, 'place_stone_0_0.pt'), 
                        'choose_set': os.path.join(rl_folder, 'choose_set_0_0.pt')}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cpu_count = 1
    mult_proc = True if cpu_count > 1 else False
    sample_games = 40
    test_games = 40
    if mult_proc:
        cpu_sample_games = sample_games // cpu_count
        cpu_test_games = test_games // cpu_count
    epochs = 100
    batch_size = 8

    n_epochs_action_net_update = 1

    env = Soulaween()
    buffer = Buffer()

    moves = ['place_stone', 'choose_set']
    value_net = {key: TargetQNet(env.get_act_space()[key]).to(device) for key in moves}
    action_net = {'place_stone': PlaceStoneNet(env.get_act_space()['place_stone']).cpu(),
                  'choose_set': ChooseSetNet(env.get_act_space()['choose_set']).cpu()}
    for key in moves:
        action_net[key].load_state_dict(value_net[key].state_dict())
        torch.save(action_net[key], saved_model_path[key])
    
    play_agent = TransformerAgent(action_net, random_action_prob=[0.1, 0.1])
    test_agent = TransformerAgent(action_net)
    random_agent = RandomAgent()
    optimizer = {key: torch.optim.Adam(value_net[key].parameters(), lr=0.0001) for key in moves}
    criterion = torch.nn.SmoothL1Loss(beta=30.0).to(device)

    score = [-10]
    print_log(f'\n{time_str()} | start training | log_path = {log_path}', log_path)
    for e in range(epochs):
        print_log('------------------------------------------------------------------------', log_path)
        print_log(f'{time_str()} | Epoch {e}', log_path) 
        print_log('SAMPLING:', log_path)
        if mult_proc:
            pool = multiprocessing.Pool(cpu_count)
            for _ in range(cpu_count):
                pool.apply_async(parallel_sampling,
                                args=(play_agent, cpu_sample_games), 
                                callback=buffer.extend)
            pool.close()
            pool.join()
        else:
            buffer.extend(parallel_sampling(play_agent, sample_games))
        n_samples = [len(buffer.memory[key]) for key in moves]
        print_log(f'{tuple(n_samples)} samples generated.', log_path)
    
    # optimize value net
        for key in moves:
            value_net[key].train()
            train_loss = []
            for step, transitions in enumerate(buffer.get_batch(key, batch_size)):
                transitions = Transition(*zip(*transitions))
                state_batch = torch.stack(transitions.state).to(device)
                reward_batch = torch.Tensor(transitions.reward).to(device) 
                # predict q values 
                q_values = torch.max(value_net[key](state_batch).squeeze(1), axis=1)[0]
                # update q network
                loss = criterion(q_values, reward_batch)
                optimizer[key].zero_grad()
                loss.backward()
                optimizer[key].step()

                train_loss.append(loss.cpu().data.item())
            print_log(f'   {key} net, {len(train_loss)} steps, loss = {np.mean(train_loss)}', log_path)
        
        buffer.clear()

        if e % n_epochs_action_net_update == 0:
            print_log('TESTING:', log_path)
            result = []
            if mult_proc:
                pool = multiprocessing.Pool(cpu_count)
                for _ in range(cpu_count):
                    pool.apply_async(parallel_arena_test, 
                                    args=(test_agent, random_agent, cpu_test_games), 
                                    callback=result.append)
                pool.close()
                pool.join()
            else:
                result.append(parallel_arena_test(test_agent, random_agent, test_games))
            s = arena_analysis(result, log_path)

            play_agent.set_action_nets(value_net)
            test_agent.set_action_nets(value_net)
            play_agent.set_random_prob(random_action_prob_scheduler(s))
            if s >= np.max(score[-20:]) or (e % 50==0):
                for key in moves:
                    file_name = os.path.join(rl_folder , f'{key}_{e}_{s}.pt')
                    torch.save(action_net[key], file_name)
                with open(os.path.join(rl_folder, f'optimizer.pickle'),'wb') as f:
                    pickle.dump(optimizer, f)
                print_log(f'   New model saved.', log_path)
            score.append(s)