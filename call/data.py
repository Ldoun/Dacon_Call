import os
import networkx as nx
import pandas as pd
from itertools import combinations_with_replacement
from torch_geometric.utils.convert import from_networkx

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


def construct_graph(feature):
    #construct graph by counselling calls & voice Mailbox Usage
    #counselling calls {0, more than 0, more than 10}
    #voice Mailbox Usage {0~1, 2, more than 2}
    
    graph = nx.Graph()
    for condition, target in [['same', 0], ['rage',[1,10]], ['over',10]]:
        node_list = get_condition_satisfied_idx(feature, '음성사서함이용', condition, target)
        graph.add_edges_from(combinations_with_replacement(node_list, 2))
    
    for condition, target in [['range', [0,1]], ['same',2], ['over',2]]:
        node_list = get_condition_satisfied_idx(feature, '상담전화건수', condition, target)
        graph.add_edges_from(combinations_with_replacement(node_list, 2))
    
    return from_networkx(graph)

def load_csv_data(path):
    train_csv = os.path.join(path, 'train.csv')
    test_csv = os.path.join(path, 'test.csv')

    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    train_x = train_data.drop(['ID', '전화해지여부'], axis = 1)
    train_y = train_data['전화해지여부']
    test_x = test_data.drop('ID', axis = 1)

    return train_x, train_y, test_x

