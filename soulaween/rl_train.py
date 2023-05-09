import torch
import numpy as np

import os
import pickle
import multiprocessing

from torch.utils.tensorboard import SummaryWriter

from soulaween.env.soulaween import Soulaween
from soulaween.utils import Buffer, print_log, time_str, parallel_sampling, Transition, random_action_prob_scheduler, parallel_arena_test, arena_analysis, get_networks, log_tensorboard
from soulaween.agents import NetworkAgent, RandomAgent

if __name__ == '__main__':
    load = False # "_570_0.4454166666666667.pt"
    load_model_folder = "first_results"
    linear = False
    model_str = 'linear' if linear else 'transformer'
    time = time_str()
    model_save_path = os.path.join('model_rl', model_str, time)
    load_model_folder = os.path.join('model_rl', model_str, load_model_folder)
    os.makedirs(model_save_path)
    log_folder = os.path.join('logs', time)
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    log_path = os.path.join(log_folder, f'{time_str()}_log_rl.txt')
    saved_model_path = {'place_stone': os.path.join(model_save_path, 'place_stone_0_0.pt'), 
                        'choose_set': os.path.join(model_save_path, 'choose_set_0_0.pt')}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir=log_folder)

    cpu_count = 1
    mult_proc = True if cpu_count > 1 else False
    sample_games = 4800
    test_games = 4800
    if mult_proc:
        cpu_sample_games = sample_games // cpu_count
        cpu_test_games = test_games // cpu_count
    epochs = 18000
    batch_size = 256

    n_epochs_action_net_update = 5

    env = Soulaween()
    obs_space = env.get_obs_space()
    act_space = env.get_act_space()
    buffer = Buffer()

    moves = ['place_stone', 'choose_set']
    action_net, value_net = get_networks(linear, load, obs_space, act_space, load_model_folder)
    
    print(sum(p.numel() for p in action_net[moves[0]].parameters() if p.requires_grad))
    
    
    play_agent = NetworkAgent(action_net, random_action_prob=[0.1, 0.1])
    test_agent = NetworkAgent(action_net)
    random_agent = RandomAgent()
    optimizer = {key: torch.optim.Adam(value_net[key].parameters(), lr=0.0005) for key in moves}
    criterion = torch.nn.SmoothL1Loss(beta=30.0)

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
                state_batch = torch.stack(transitions.state)
                reward_batch = torch.tensor(transitions.reward)
                action_batch = torch.LongTensor(transitions.action).unsqueeze(1)
                # predict q values 
                q_values = value_net[key](state_batch).gather(1, action_batch).flatten()
                # update q network
                loss = criterion(q_values, reward_batch)
                optimizer[key].zero_grad()
                loss.backward()
                optimizer[key].step()

                train_loss.append(loss.data.item())
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

            if s >= np.max(score) or (e % 50==0):
                for key in moves:
                    file_name = os.path.join(model_save_path , f'{key}_{e}_{s:.2f}.pt')
                    torch.save(action_net[key], file_name)
                with open(os.path.join(model_save_path, f'optimizer.pickle'),'wb') as f:
                    pickle.dump(optimizer, f)
                print_log(f'   New model saved.', log_path)

            play_agent.set_action_nets(value_net)
            test_agent.set_action_nets(value_net)
            play_agent.set_random_prob(random_action_prob_scheduler(s))
            
            score.append(s)

            epoch_stats = {
                'epoch': e+1,
                'epoch_score': s,
            }
            log_tensorboard(writer, epoch_stats)
            
    writer.close()
