import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing

from soulaween.env.soulaween import Soulaween
from soulaween.utils.utils import error


def test_range(cpu_index=None, cpu_count=None):
    num_states = 3**16
    if cpu_index is not None:
        width = np.ceil(num_states / cpu_count)
        cpu_range = np.clip(np.arange(cpu_index * width, (cpu_index + 1) * width), a_min=None, a_max=num_states)
    else:
        cpu_range = np.arange(num_states)

    env = Soulaween()
    env.reset()

    cpu_dict = dict()

    for state_ind in tqdm(cpu_range):
        empty = np.zeros(16, dtype=np.int8)
        cross = np.zeros(16, dtype=np.int8)
        circle = np.zeros(16, dtype=np.int8)

        # create board state from state index
        rest = state_ind
        for i in range(16):
            field = rest % 3
            if field == 0:
                empty[i] = 1
            elif field == 1:
                cross[i] = 1
            elif field == 2:
                circle[i] = 1
            rest //= 3
        state = [empty, cross, circle]
        env.set_board(state)

        # try all legal actions
        mask = env.legal_actions
        legal_actions = np.argwhere(mask==1).flatten()
        good_act = []
        for action in legal_actions:
            env.reset()
            env.set_board(state)
            _, _, done, _ = env.step(action)
            if done:
                good_act.append(action)
        
        # append legal actions to dict
        if not len(good_act) == 0:
            cpu_dict[state_ind] = good_act
    print(f'cpu {cpu_index} done, processed {cpu_range[-1] - cpu_range[0]} items')
    return cpu_dict


if __name__ == "__main__":
    cpu_count = 24
    mult_proc = True if cpu_count > 1 else False

    set_dict = dict()
    if mult_proc:
        with multiprocessing.Pool(cpu_count) as pool:
            for i in range(cpu_count):
                pool.apply_async(test_range, 
                                args=(i, cpu_count), 
                                callback=set_dict.update,
                                error_callback=error)
            pool.close()
            pool.join()
    else:
        set_dict.update(test_range())

    mat = np.zeros((3**16, 32), dtype=bool)
    for state_ind, acts in tqdm(set_dict.items()):
        for a in acts:
            mat[int(state_ind), a] = True
    print('created numpy array')
    np.savez_compressed('complete_set.npz', mat)

    print('done')
