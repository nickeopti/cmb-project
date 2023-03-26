import os
import random
import sys

import yaml

if __name__ == '__main__':
    file_names = sys.argv[1:]
    base_file_names = [os.path.basename(file_name)[:-7] for file_name in file_names]

    random.shuffle(base_file_names)

    frac_train, frac_val = 0.7, 0.2
    n = len(base_file_names)
    n_train = round(n * frac_train)
    n_val = round(n * frac_val)

    train = base_file_names[:n_train]
    val = base_file_names[n_train:n_train + n_val]
    test = base_file_names[n_train + n_val:]

    with open('split.yaml', 'w') as f:
        yaml.dump({'train': train, 'val': val, 'test': test}, f)
