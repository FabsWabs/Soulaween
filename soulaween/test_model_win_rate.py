import multiprocessing
import os

from soulaween.env.soulaween import Soulaween
from soulaween.utils.utils import print_log, parallel_arena_test, arena_analysis, seed_everything, get_networks
from soulaween.agents import NetworkAgent, RandomAgent, RuleBased


if __name__ == "__main__":
    seed_everything(42)
    load = ".pt"
    load_model_folder = "best"
    linear = True
    model_str = 'linear' if linear else 'transformer'
    load_model_folder = os.path.join('model_rl', model_str, load_model_folder)

    cpu_count = 1
    mult_proc = True if cpu_count > 1 else False
    log_path = None
    rl_folder = os.path.join('model_rl', model_str)#, "20230504-200525")

    test_games = 4800
    if mult_proc:
        cpu_test_games = test_games // cpu_count
    
    moves = ['place_stone'] #, 'choose_set']
    with_choose_set = True if len(moves) == 2 else False

    env = Soulaween(with_choose_set=with_choose_set)
    obs_space = env.get_obs_space()
    act_space = env.get_act_space()

    action_net, value_net = get_networks(linear, load, obs_space, act_space, load_model_folder, with_choose_set=with_choose_set)
    test_agent = NetworkAgent(action_net)
    random_agent = RuleBased()
    # random_agent = RandomAgent()


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