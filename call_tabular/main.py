import importlib
import pandas as pd

from sklearn.model_selection import StratifiedKFold

import torch

from trainer import Trainer
from config import get_config
from utils import seed_everything
from data import load_csv_data, load_data_loader

if __name__ == "__main__":
    args = get_config()
    print(args)

    seed_everything(args.seed)
    model_module = importlib.import_module("model")
    model = getattr(model_module, args.model)

    train_x, train_y, test_x = load_csv_data(args.raw_path)

    skf = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model(
        input_size=train_x.shape[1], hidden_size=args.hidden, output_size=1, drop_p=args.drop_p
    ).to(device)
    print(model)

    test_predictoins = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)):
        model.reset_parameters()        
        train_tensor_x = torch.tensor(train_x.values, dtype=torch.float, device=device)
        train_tensor_y = torch.tensor(train_y.values, dtype=torch.float, device=device).unsqueeze(-1)

        train_loader = load_data_loader(args,train_tensor_x[train_idx], train_tensor_y[train_idx],is_train=True)
        valid_loader = load_data_loader(args,train_tensor_x[valid_idx], train_tensor_y[valid_idx],is_train=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        pos_weight = [(train_y.loc[train_idx]==0).sum()/train_y.loc[train_idx].sum()]
        print(f"Using pos_weight of {pos_weight} for positive classs")
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float, device=device))
        
        trainer = Trainer(args, train_loader, valid_loader, model, optimizer, loss_fn)
        trainer.train()

        ####Testing####

    