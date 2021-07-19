import numpy as np

def load_npy():
    return (np.load('./datasets/X_train.npy'), np.load('./datasets/Y_train.npy')), (np.load('./datasets/X_test.npy'), np.load('./datasets/Y_test.npy'))
