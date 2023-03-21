import os
import pandas as pd

import torch
import torch.nn.functional as f
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader



def get_dataloader(features, edge_index, num_neighbors, batch_size):
    data = Data(
        x = features,
        edge_index = edge_index,
    )

    loader = NeighborLoader(
        data = data,
        num_neighbors = num_neighbors, #[30] * 2 need to be bigger?
        batch_size = batch_size,
        shuffle=True,
        directed = False,
    )

    return loader

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

def load_csv_data(path):
    train_csv = os.path.join(path, 'train.csv')
    test_csv = os.path.join(path, 'test.csv')

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_x = train_data.drop(['ID', '전화해지여부'], axis = 1)
    train_y = train_data['전화해지여부']
    test_x = test_data.drop('ID', axis = 1)

    return train_x, train_y, test_x

