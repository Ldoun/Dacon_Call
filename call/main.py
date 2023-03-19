import importlib
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import torch

from config import get_config
from utils import seed_everything
from data import load_csv_data, construct_graph

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
        train_graph = construct_graph(train_x.loc[train_idx, :], device)
        valid_graph = construct_graph(train_x, device)

        # optimizer = torch.optim.Adam([
        #     dict(params=model.conv1.parameters(), weight_decay=5e-4),
        #     dict(params=model.conv2.parameters(), weight_decay=0)
        # ], lr=args.lr) #GCN model optimizer

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        pos_weight = [(train_y.loc[train_idx]==0).sum()/train_y.loc[train_idx].sum()]
        print(f"Using pos_weight of {pos_weight} for positive classs")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(pos_weight))
        
        train_tensor_x = torch.FloatTensor(train_x.values, device=device)
        train_tensor_y = torch.FloatTensor(train_y.values, device=device)
        for epoch in range(1, args.epochs+1):
            model.train()
            train_out = model(train_tensor_x[train_idx], train_graph)
            train_loss = loss_fn(train_out, train_tensor_y[train_idx]) #compute loss for train set, graph(only with train node)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_f1_score = f1_score(train_y.loc[train_idx], train_out.detach().cpu().numpy())

            model.eval()
            with torch.no_grad():
                valid_out = model(train_tensor_x, valid_graph)
                valid_loss = loss_fn(valid_out[valid_idx], train_tensor_y[valid_idx]) #compute loss for valid set, graph(with train&valid node)
                valid_f1_score = f1_score(train_y.loc[valid_idx], valid_out[valid_idx].detach().cpu().numpy())
            
            print(f"{epoch}-epoch: t_loss {train_loss.item()} t_f1 {train_f1_score} v_loss {valid_loss.item()} v_f1 {valid_f1_score}")

        model.eval()
        test_graph = construct_graph(all_x, device)
        with torch.no_grad():
            test_predictoins.append(model(torch.FloatTensor(all_x, device=device), test_graph).detach().cpu().numpy())

