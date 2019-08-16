import numpy as np
import os

def find_nearest_idx(array, value):
    # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


class NPYSelector():
    def __init__(self, path, idx=None):
        self.data = np.load(os.path.join(os.path.realpath(__file__), path))
        if idx is not None:
            self.idx = idx
        else:
            self.idx = None

    def sel(self, idx_val, method):
        assert method == "nearest", "not implemented!"
        assert self.idx is not None, "no index"
        idx = find_nearest_idx(self.idx, idx_val)
        return self.data[idx]


class NPYDataset():
    def __init__(self):
        self.time = NPYSelector("time.npy")
        self.X = NPYSelector("X.npy",
                             idx="time.npy")
