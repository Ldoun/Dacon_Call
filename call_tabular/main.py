import importlib
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


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
    model_module = getattr(model_module, args.model)

    train_x, train_y, test_x = load_csv_data(args.raw_path)
    scaler = MinMaxScaler()
    scaler.fit(train_x.values)
    
    train_x = scaler.transform(train_x.values)
    test_x = scaler.transform(test_x.values)
    train_y = train_y.values

    skf = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_predictoins = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)):
        print(f'---------{fold_idx}-fold-------------')
        model = model_module(
            input_size=train_x.shape[1], hidden_size=args.hidden, output_size=1, n_layer=args.n_layer, drop_p=args.drop_p
        ).to(device)
        print(model)

        train_loader = load_data_loader(
            args=args, data=train_x[train_idx], label=train_y[train_idx], is_train=True,device=device,use_oversample=args.oversampling)
        valid_loader = load_data_loader(
            args=args, data=train_x[valid_idx], label=train_y[valid_idx], is_train=True, device=device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if args.weighted_loss:
            weight = [1 - (train_y[train_idx]==0).sum()/len(train_idx), 1 - (train_y[train_idx]==1).sum()/len(train_idx) ]
            print(f"Using weight of {weight}")
            loss_fn = torch.nn.BCELoss(weight=torch.tensor(weight, dtype=torch.float, device=device))
        else:
            loss_fn = torch.nn.BCELoss()
        
        trainer = Trainer(args, train_loader, valid_loader, model, optimizer, loss_fn)
        trainer.train()

        ####Testing####

    