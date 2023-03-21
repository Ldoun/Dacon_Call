import os
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler

class TabularDataset(Dataset):
    def __init__(self,data,label,is_train, use_oversample, device) -> None:
        self.data = data
        self.label = label
        self.is_train = is_train
        self.device = device
        if use_oversample:
            ros = RandomOverSampler()
            self.data, self.label = ros.fit_resample(data, label)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float, device=self.device)
        if self.is_train:
            y = torch.tensor(self.label[idx], dtype=torch.float, device=self.device).unsqueeze(-1)
            return {"x":x, "y":y}
        else:
            return x
    

def load_data_loader(args, data, device, label=None,is_train=False,use_oversample=False):
    dataset = TabularDataset(data, label, is_train, use_oversample=use_oversample, device=device)
    dataloader = DataLoader(dataset, args.batch_size, shuffle=is_train)
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

