import torch
from tqdm import tqdm

from soulaween.utils.utils import compute_state_ind


for state_ind in tqdm(range(3**16)):
    empty = torch.zeros(16, dtype=torch.int8)
    cross = torch.zeros(16, dtype=torch.int8)
    circle = torch.zeros(16, dtype=torch.int8)

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
    state_ind_val = compute_state_ind(state)

    if state_ind_val != state_ind:
        print('Error in state index computation')
        break