import torch
import numpy as np
import torch.utils.data as data
import h5py
from utils.tools import draw
from torch.utils.data import DataLoader
# References:
# - https://github.com/paul007pl/MVP_Benchmark/tree/main/completion

class MVP(data.Dataset):
    def __init__(self, config, args=None):
        self.subset = config.subset
        if self.subset=="train":
            self.file_path = f'{config.pc_path}/Completion/MVP_Train_CP.h5'
        elif self.subset == "val" or self.subset=="test":
            self.file_path = f'{config.pc_path}/Completion/MVP_Test_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        print('-- Info')
        print(f'[DATASET] Open file {self.file_path}')
        print(f'[DATASET] {self.input_data.shape[0]} instances were loaded')

        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])
        input_file.close()
        self.len = self.input_data.shape[0]
        self.indices = list(range(self.len))
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.subset == "train":
            np.random.shuffle(self.indices)
        idx = self.indices[index]
        partial = torch.from_numpy(self.input_data[idx])
        complete = torch.from_numpy(self.gt_data[idx//26])
        label = self.labels[idx].sum()
        sample = {'gt': complete, 'partial': partial, 'taxonomy_id': label}
        return sample
