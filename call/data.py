import os
import numpy as np
from scipy.sparse import coo_array
import pandas as pd
from itertools import combinations_with_replacement
from torch_geometric.utils.convert import from_scipy_sparse_matrix

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


def construct_graph(feature):
    #construct graph by counselling calls & voice Mailbox Usage
    #counselling calls {0, more than 0, more than 10}
    #voice Mailbox Usage {0~1, 2, more than 2}
    
    row = np.array()
    col = np.array()
    for condition, target in [['same', 0], ['range',[1,10]], ['over',10]]:
        node_list = get_condition_satisfied_idx(feature, '음성사서함이용', condition, target)
        t_row, t_col = zip(*combinations_with_replacement(node_list, 2))
        row = np.append(row, np.array(t_row))
        col = np.append(col, np.array(t_col))
        print(f'음성사서함이용 {condition} {target} edge added')
    
    for condition, target in [['range', [0,1]], ['same',2], ['over',2]]:
        node_list = get_condition_satisfied_idx(feature, '상담전화건수', condition, target)
        t_row, t_col = zip(*combinations_with_replacement(node_list, 2))
        row = np.append(row, np.array(t_row))
        col = np.append(col, np.array(t_col))
        print(f'상담전화건수 {condition} {target} edge added')
    

    sparse_matrix = coo_array((np.ones(len(row), dtype=bool),(row, col)))
    return from_scipy_sparse_matrix(sparse_matrix)

def load_csv_data(path):
    train_csv = os.path.join(path, 'train.csv')
    test_csv = os.path.join(path, 'test.csv')

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_x = train_data.drop(['ID', '전화해지여부'], axis = 1)
    train_y = train_data['전화해지여부']
    test_x = test_data.drop('ID', axis = 1)

    return train_x, train_y, test_x

