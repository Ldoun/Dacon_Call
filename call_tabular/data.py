import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler

class TabularDataset(Dataset):
    def __init__(self,data,label,is_train, use_oversample) -> None:
        self.is_train = is_train

        if use_oversample:
            ros = RandomOverSampler()
            data, label = ros.fit_resample(data, label)
            print('using over sampled data')

        if self.is_train:
            self.label = torch.tensor(label, dtype=torch.float)
        self.data = torch.tensor(data, dtype=torch.float)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_train:
            return {"x":self.data[idx], "y":self.label[idx]}
        else:
            return self.data[idx]
    

def load_data_loader(args, data, label=None,is_train=False,use_oversample=False):
    dataset = TabularDataset(data, label, is_train, use_oversample=use_oversample)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=is_train, pin_memory=is_train, num_workers=2)
    return dataloader

def load_csv_data(path):
    train_csv = os.path.join(path, 'train.csv')
    test_csv = os.path.join(path, 'test.csv')

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_x = train_data.drop(['ID', '전화해지여부'], axis = 1)
    train_y = train_data['전화해지여부']
    test_x = test_data.drop('ID', axis = 1)

    return train_x, train_y, test_x

