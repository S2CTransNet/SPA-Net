import torch.utils.data as data
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
import os
from torch.utils.data import DataLoader
import numpy as np


class KITTI(data.Dataset):
    def __init__(self, config, arg=None):
        self.npoints = config.n_points
        self.subset = config.subset
        self.npy_file_list = self._load_npy_file_list(config.txt_path)
        self._get_file_list(self.npy_file_list)
        self.transforms = self._get_transforms(self.subset)

    def _load_npy_file_list(self, category_file):
        with open(category_file, 'r') as f:
            lines = f.readlines()
            file_list = [line.strip() for line in lines]
        return file_list

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            },{
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def _get_file_list(self, npy_file_list, n_renderings=1):
        """Prepare file list for the dataset"""
        self.data = []
        for line in npy_file_list:
            path = f'{os.path.dirname(BASE_DIR)}/data/KITTI/{line}'
            self.data.append(np.load(path)[:,:3])
        print(f'[DATASET] {len(self.data)} {self.subset}ing instances were loaded')

    def pc_norm(self, pc):

        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc


    def __getitem__(self, idx):
        data = {}
        data['partial'] = self.pc_norm(self.data[idx])
        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def __len__(self):
        return len(self.data)

# class Config:
#     def __init__(self):
#         self.n_points = 8192  # 示例值
#         self.subset = "test"  # 示例值
#         self.txt_path = "/home/t5820/yal/code_5.0/dataloader/KITTI/test/500-1500.txt"
#
#
#
# config = Config()
# dataloder = KITTI(config)
# data_loader = DataLoader(dataloder, batch_size=1, shuffle=False)
# for sample in data_loader:
#     print(sample)