import numpy as np
import pickle
from tqdm import tqdm
import numpy as np

with open('complete_set.pickle', 'rb') as handle:
    set_dict = pickle.load(handle)

print('loaded dict')

better_dict = {int(x): y for x, y in set_dict.items()}

print('converted dict')

with open('complete_set_int.pickle', 'wb') as handle:
    pickle.dump(set_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('saved new dict')

mat = np.zeros((3**16, 32), dtype=bool)
for state_ind, acts in tqdm(better_dict.items()):
    for a in acts:
        mat[state_ind, a] = True

print('created numpy array')

np.savez_compressed('complete_set.npz', mat)

print('saved .npz file')

mat = np.load('complete_set.npy')
comp = np.load('complete_set.npz')['a']

print(np.array_equal(mat, comp))