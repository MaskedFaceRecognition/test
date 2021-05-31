import os
import numpy as np
# clear[201210318]
# to_num.py
def load_data(dir_):
    x_train = np.load(os.path.join(dir_, 'x_train.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test.npy'))
    return x_train, x_test

if __name__ == '__main__':
    x_train, x_test = load_data()
    print(x_train.shape)
    print(x_test.shape)
