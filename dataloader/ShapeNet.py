import os
import torch
import numpy as np
import torch.utils.data as data
import h5py
import open3d
import random


class ShapeNet(data.Dataset):
    def __init__(self, config, args=None):
        self.data_root = config.data_path
        self.pc_path = config.pc_path
        self.subset = config.subset
        self.npoints = config.n_points
        if self.subset == 'train':
            self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
            self.load_data_list(self.data_list_file)
            self.split_train_val()
            self.file_list_ = self.train_file_list
            print('-- Info')
            print(f'[DATASET] Open file {self.data_list_file}')
            print(f'[DATASET] {len(self.train_file_list)} training instances were loaded')
        elif self.subset == 'test' or 'val':
            self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
            self.load_data_list(self.data_list_file)
            self.file_list_ = self.file_list
            print(f'[DATASET] Open file {self.data_list_file}')
            print(f'[DATASET] {len(self.file_list_)} {self.subset} instances were loaded')

    def load_data_list(self, file_path):
        self.file_list = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        if self.subset == 'train':
            random.shuffle(self.file_list)

    def split_train_val(self):
        taxonomy_dict = {}
        for item in self.file_list:
            if item['taxonomy_id'] not in taxonomy_dict:
                taxonomy_dict[item['taxonomy_id']] = []
            taxonomy_dict[item['taxonomy_id']].append(item)
        self.val_file_list = []
        self.train_file_list = []
        for taxonomy_id, items in taxonomy_dict.items():
            n = min(int(len(items)/5),20)
            random.shuffle(items)
            val_items = items[:n]
            train_items = items[n:]
            self.val_file_list.extend(val_items)
            self.train_file_list.extend(train_items)
        val_file_paths = [item['file_path'] for item in self.val_file_list]
        with open(os.path.join(self.data_root, 'val.txt'), 'w') as f:
            for path in val_file_paths:
                f.write(f'{path}\n')

    def pc_norm(self, pc):

        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list_[idx]
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        sample = {'gt': data, 'taxonomy_id': sample['taxonomy_id'],
                  'model_id': sample['model_id']}
        return sample
    def __len__(self):
        return len(self.file_list_)

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

