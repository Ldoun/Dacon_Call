import os
import pandas as pd

import torch
import torch.nn.functional as f
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from utils import print_variable

def load_csv_data(path):
    train_csv = os.path.join(path, 'train.csv')
    test_csv = os.path.join(path, 'test.csv')

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_x = train_data.drop(['ID', '전화해지여부'], axis = 1)
    train_y = train_data['전화해지여부']
    test_x = test_data.drop('ID', axis = 1)

    return train_x, train_y, test_x

def load_data_loader(args, data, mask=None, label=None, device='cpu', shuffle=False):
    data = torch.tensor(data, dtype=torch.float, device=device)
    if label is not None:
        label = torch.tensor(label, dtype=torch.float, device=device).unsqueeze(-1)
    edge_index = construct_graph(data, device=device)
    dataloader = get_dataloader(data, edge_index, mask=mask, label=label, num_neighbors=[args.num_neighbor] * args.num_hop, batch_size=args.batch_size, shuffle=shuffle)

    return dataloader

def construct_graph(feature, device):
    cordinate = torch.tensor([], dtype=torch.long, device=device)
    for index, sample in enumerate(feature):
        similarity = f.cosine_similarity(sample, feature, dim=1)
        similarity[index] = 0.
        _, top_similar_samples = similarity.topk(5)
        row = torch.tensor([index]*len(top_similar_samples), dtype=torch.long, device=device)
        temp = torch.stack([row, top_similar_samples],dim=1)
        cordinate = torch.concat([cordinate, temp], dim=0)

    return cordinate.T

def get_dataloader(features, edge_index, mask, num_neighbors, batch_size, label=None, shuffle=False):
    if label is not None:
        print_variable({'x' : features, 'edge' : edge_index, 'label':label})
    else:
        print_variable({'x' : features, 'edge' : edge_index})

    data = Data(
        x = features,
        edge_index = edge_index,
        y = label,
    )

    loader = NeighborLoader(
        data = data,
        num_neighbors = num_neighbors, #[30] * 2
        input_nodes = mask,
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers = 2, 
    )

    return loader
