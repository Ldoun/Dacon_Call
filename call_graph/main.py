import importlib
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch

from trainer import Trainer
from config import get_config
from utils import seed_everything
from data import load_csv_data, construct_graph, get_dataloader

if __name__ == "__main__":
    args = get_config()
    print(args)

    seed_everything(args.seed)
    model_module = importlib.import_module("torch_geometric.nn.models")
    model = getattr(model_module, args.model)

    train_x, train_y, test_x = load_csv_data(args.raw_path)
    all_x = pd.concat([train_x, test_x], axis = 0)

    skf = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model(
        in_channels=train_x.shape[1], hidden_channels=args.hidden, num_layers=2, out_channels=1, dropout=args.drop_p
    ).to(device)
    print(model)

    test_predictoins = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)):
        model.reset_parameters()
        
        tensor_valid_idx = torch.tensor(valid_idx, dtype=torch.long, device=device)
        train_tensor_x = torch.tensor(train_x.values, dtype=torch.float, device=device)
        train_tensor_y = torch.tensor(train_y.values, dtype=torch.float, device=device).unsqueeze(-1)
        
        graph = construct_graph(train_tensor_x, device)
        train_index = (torch.isin(graph[0,:], tensor_valid_idx) | torch.isin(graph[1,:], tensor_valid_idx)) == False

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        pos_weight = [(train_y.loc[train_idx]==0).sum()/train_y.loc[train_idx].sum()]
        print(f"Using pos_weight of {pos_weight} for positive classs")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float, device=device))
        
        train_loader = get_dataloader(train_tensor_x[train_idx], graph[:, train_index], num_neighbors=[args.num_neighbors] * 2, batch_size=args.batch_size)
        valid_loader = get_dataloader(train_tensor_x, graph, num_neighbors=[args.num_neighbors] * 2, batch_size=args.batch_size,input_nodes=tensor_valid_idx)

        trainer = Trainer(args, train_loader, valid_loader, model, optimizer, loss_fn)
        trainer.train()

    