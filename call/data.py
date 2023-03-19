import os
import pandas as pd

import torch
from torch import combinations, sparse_coo_tensor

def get_condition_satisfied_idx(data, col_name, condition, target):
    feat = data[col_name]
    if condition == 'same':
        return feat[feat == target].index
    elif condition == 'over':
        return feat[feat > target].index
    elif condition == 'under':
        return feat[feat < target].index
    elif condition == 'range':
        return feat[(feat >= target[0]) & (feat <= target[1])].index
    else:
        raise "condition not satisfied"


def construct_graph(feature, device):
    #construct graph by counselling calls & voice Mailbox Usage
    #counselling calls {0, more than 0, more than 10}
    #voice Mailbox Usage {0~1, 2, more than 2}
    
    cordinate = torch.tensor([], device=device,dtype=torch.long)
    for target in range(0, max(feature['음성사서함이용'])+1):
        node_list = get_condition_satisfied_idx(feature, '음성사서함이용', 'same', target)
        cordinate = torch.concat([combinations(torch.tensor(node_list, device=device,dtype=torch.long), 2), cordinate], axis=0)
        print(f'음성사서함이용 same {target} -> edge added')
    print(f'added {cordinate.shape[0]} edges')
    
    for target in range(0, max(feature['상담전화건수'])+1):
        node_list = get_condition_satisfied_idx(feature, '상담전화건수', 'same', target)
        cordinate = torch.concat([combinations(torch.tensor(node_list, device=device,dtype=torch.long), 2), cordinate], axis=0)
        print(f'상담전화건수 same {target} -> edge added')
    
    #sparse_tensor = sparse_coo_tensor(cordinate.T, torch.ones(cordinate.shape[0], dtype=bool, device=device))
    print(f'added {cordinate.shape[0]} edges')
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

