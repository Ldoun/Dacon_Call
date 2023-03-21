import importlib
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler


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

    test_predictoins = []
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)):
        print(f'---------{fold_idx}-fold-------------')
        model = model(
            input_size=train_x.shape[1], hidden_size=args.hidden, output_size=1, drop_p=args.drop_p
        ).to(device)

        ros = RandomOverSampler()
        oversampled_data, oversampled_label = ros.fit_resample(train_x.loc[train_idx], train_y.loc[train_idx])
        train_tensor_x = torch.tensor(oversampled_data.values, dtype=torch.float, device=device)
        train_tensor_y = torch.tensor(oversampled_label.values, dtype=torch.float, device=device).unsqueeze(-1)

        valid_tensor_x = torch.tensor(train_x.loc[valid_idx].values, dtype=torch.float, device=device)
        valid_tensor_y = torch.tensor(train_y.loc[valid_idx].values, dtype=torch.float, device=device).unsqueeze(-1)

        train_loader = load_data_loader(args,train_tensor_x, train_tensor_y,is_train=True)
        valid_loader = load_data_loader(args, valid_tensor_x, valid_tensor_y,is_train=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # pos_weight = [(train_y.loc[train_idx]==0).sum()/train_y.loc[train_idx].sum()]
        # print(f"Using pos_weight of {pos_weight} for positive classs")
        loss_fn = torch.nn.BCEWithLogitsLoss()# pos_weight=torch.tensor(pos_weight, dtype=torch.float, device=device))
        
        trainer = Trainer(args, train_loader, valid_loader, model, optimizer, loss_fn)
        trainer.train()

        ####Testing####

    