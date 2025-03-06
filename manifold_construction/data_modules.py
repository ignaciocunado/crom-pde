import math
import os
import random
import re

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

'''
Simulation DataModule
'''


class ManifoldConstructionDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 16, num_workers=2):
        super().__init__()
        self.data_path = 'data/thermodynamics/output/0'
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_list = DataList(self.data_path, 1.0)
        assert (len(self.data_list.data_list) > 0)

        self.sim_dataset = ManifoldConstructionDataset(self.data_path, self.data_list.data_list)
        self.getMinMaxX()
        self.getMeanStdQ()
        self.getMeanStdX()

    def train_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True)

    def getMeanStdQ(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_q.npy')

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError('Upload the mean and STD for Q')

        with open(preprocessed_file, 'rb') as f:
            self.mean_q = np.load(f)
            self.std_q = np.load(f)

    def getMeanStdX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_x.npy')

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError('Upload the mean and STD for X')

        with open(preprocessed_file, 'rb') as f:
            self.mean_x = np.load(f)
            self.std_x = np.load(f)

    def getMinMaxX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_x.npy')

        if not os.path.exists(preprocessed_file):
            raise FileNotFoundError('Upload the min and max values for ')

        with open(preprocessed_file, 'rb') as f:
            self.min_x = np.load(f)
            self.max_x = np.load(f)

    def get_dataParams(self, ):
        return {'mean_q': self.mean_q, 'std_q': self.std_q, 'mean_x': self.mean_x, 'std_x': self.std_x,
                'min_x': self.min_x, 'max_x': self.max_x}

    def get_dataFormat(self, ):
        example_input_array = torch.unsqueeze(self.sim_dataset[0]['encoder_input'], 0)
        [_, i_dim] = self.sim_dataset[0]['x'].shape
        [npoints, o_dim] = self.sim_dataset[0]['q'].shape

        data_format = {'i_dim': i_dim, 'o_dim': o_dim, 'npoints': npoints, 'data_path': self.data_path}

        return data_format, example_input_array


'''
Simulation Dataset
'''


class ManifoldConstructionDataset(Dataset):
    def __init__(self, data_path, data_list):
        self.data_list = data_list
        self.data_path = data_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        filename = self.data_list[idx]
        sim_data = DataState(filename)

        x = sim_data.x[:]
        q = sim_data.q[:]
        time = sim_data.t

        x = torch.from_numpy(x).float()
        q = torch.from_numpy(q).float()

        encoder_input = torch.cat((q, x), 1)

        data_item = {'filename': sim_data.filename, 'x': x,
                     'q': q,
                     'encoder_input': encoder_input, 'time': time}

        return data_item


'''
Simulation State
'''


class DataState(object):
    def __init__(self, filename, readfile=True, input_x=None, input_q=None, input_t=None, label=None):
        self.filename = filename
        if readfile:
            with h5py.File(self.filename, 'r') as h5_file:
                self.x = h5_file['/x'][:]
                self.x = np.array(self.x.T)
                self.q = h5_file['/q'][:]
                self.q = np.array(self.q.T)
                self.t = h5_file['/time'][0][0]
        else:
            if input_x is None:
                print('must provide a x if not reading from file')
                exit()
            if input_q is None:
                print('must provide a q if not reading from file')
                exit()
            if input_t is None:
                print('must provide a t if not reading from file')
                exit()
            self.x = input_x
            self.q = input_q
            self.t = input_t
            self.label = label

    def write_to_file(self, filename=None):
        if filename:
            self.filename = filename
        print('writng sim state: ', self.filename)
        dirname = os.path.dirname(self.filename)
        os.umask(0)
        os.makedirs(dirname, 0o777, exist_ok=True)
        with h5py.File(self.filename, 'w') as h5_file:
            dset = h5_file.create_dataset("x", data=self.x.T)
            dset = h5_file.create_dataset("q", data=self.q.T)
            self.t = self.t.astype(np.float64)
            dset = h5_file.create_dataset("time", data=self.t)
            if self.label is not None:
                label = self.label.reshape(-1, 1)
                label = label.astype(np.float64)
                dset = h5_file.create_dataset("label", data=label)


class DataList(object):
    def __init__(self, root_dir, train_ratio):
        self.data_list, self.data_train_list, self.data_test_list, self.data_train_dir, self.data_test_dir = obtainFilesRecursively(
            root_dir, train_ratio)


def obtainFilesRecursively(path, train_ratio):
    config_file_pattern = r'h5_f_(\d+)\.h5'
    config_file_matcher = re.compile(config_file_pattern)
    dir_pattern = r'sim_seq_(.*?)'
    dir_matcher = re.compile(dir_pattern)

    data_list = []
    data_train_list = []
    data_test_list = []
    data_train_dir = []
    data_test_dir = []

    dir_list = os.listdir(path)

    num_sims = 0
    dir_list_sim = []
    for dirname in dir_list:
        if os.path.isdir(os.path.join(path, dirname)):
            dir_match = dir_matcher.match(
                dirname)
            if dir_match != None:
                num_sims += 1
                dir_list_sim.append(dirname)
    random.seed(0)
    random.shuffle(dir_list_sim)

    train_size = math.ceil(train_ratio * num_sims)
    test_size = num_sims - train_size

    counter = 0
    for dirname in dir_list_sim:
        data_list_local = data_train_list if counter < train_size else data_test_list
        data_dir_local = data_train_dir if counter < train_size else data_test_dir
        data_dir_local.append(os.path.join(path, dirname))
        counter += 1
        for filename in os.listdir(os.path.join(path, dirname)):
            config_file_match = config_file_matcher.match(
                filename)
            if config_file_match is None:
                continue
            # skip files begin
            file_number = int(config_file_match[1])
            # skip files finish
            # print(file_number)
            fullfilename = os.path.join(path, dirname, filename)
            data_list.append(fullfilename)
            data_list_local.append(fullfilename)
        # exit()
    return data_list, data_train_list, data_test_list, data_train_dir, data_test_dir
