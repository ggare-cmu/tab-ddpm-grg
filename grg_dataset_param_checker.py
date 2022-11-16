import numpy as np


if __name__ == "__main__":
    train_idx = "/home/grg/Research/tab-ddpm-grg/data/higgs-small/idx_train.npy"
    train_idx = np.load(train_idx)

    train_X = "/home/grg/Research/tab-ddpm-grg/data/higgs-small/X_num_train.npy"
    train_X = np.load(train_X)

    train_y = "/home/grg/Research/tab-ddpm-grg/data/higgs-small/y_train.npy"
    train_y = np.load(train_y)

    train_idx