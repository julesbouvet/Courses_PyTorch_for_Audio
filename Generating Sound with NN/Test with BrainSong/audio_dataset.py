import numpy as np
import torch
from torch.utils.data import Dataset

class AudioDataSet(Dataset):

    def __init__(self, npz_file):
        self.all_sample = torch.Tensor(np.load(npz_file)['arr_0'])

    def __len__(self):
        return self.all_sample.shape[0]

    def __getitem__(self, item):
        return self.all_sample[item]






