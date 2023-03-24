import os
import importlib
import numpy as np
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
    model_module = importlib.import_module("torch_geometric.nn.models")
    model_module = getattr(model_module, args.model)

    trial = str(len(os.listdir('./model_file')))
    model_path = os.path.join('model_file', trial)

    os.makedirs(model_path, exist_ok=True)
    print(f'saving model to {model_path}')
    test_file = f'{trial}.csv'
    print(f'saving test result to {test_file}')

    train_x, train_y, test_x = load_csv_data(args.raw_path)
    scaler = MinMaxScaler()
    scaler.fit(train_x.values)
    
    train_x = scaler.transform(train_x.values)
    test_x = scaler.transform(test_x.values)
    train_y = train_y.values

    all_x = np.concatenate([train_x, test_x], axis=0)
    test_idx = np.array(range(train_x.shape[0], all_x.shape[0]))

    if args.stacking_file is not None:
        print(f'train_x shape: {train_x.shape}')
        train_prediction = pd.read_csv(os.path.join(args.raw_path, args.stacking_file+'train.csv'))
        test_prediction = pd.read_csv(os.path.join(args.raw_path, args.stacking_file+'test.csv'))
        train_x = np.concatenate([train_x,train_prediction.values],axis=1)
        test_x = np.concatenate([test_x,test_prediction.values],axis=1)
        print(f'stacking train_x shape: {train_x.shape}')

    skf = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_prediction = pd.DataFrame()
    validation_f1 = []

    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(train_x, train_y)):
        print(f'---------{fold_idx}-fold-------------')
        model = model_module(
            input_size=train_x.shape[1], hidden_size=args.hidden, output_size=1, n_layer=args.n_layer, drop_p=args.drop_p
        ).to(device)
        
        if fold_idx == 0:
            print(model)

        train_loader = load_data_loader(
            args=args, data=train_x[train_idx], label=train_y[train_idx], is_train=True, use_oversample=args.oversampling) #only train
        valid_loader = load_data_loader(
            args=args, data=train_x, label=train_y, mask=valid_idx, is_train=True) #train + valid
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if args.weighted_loss:
            weight = [(1 - (train_y[train_idx]==1).sum()/len(train_idx))*10]
            print(f"Using weight of {weight}")
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, dtype=torch.float, device=device))
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        trainer = Trainer(args, train_loader, valid_loader, model, optimizer, loss_fn, device, model_path)
        validation_f1.append(trainer.train())

        test_loader = load_data_loader(args=args, data=all_x, mask=test_idx, is_train=False) #train + valid + test
        test_prediction[f'{fold_idx}-fold'] = pd.Series(trainer.test(test_loader))
        test_prediction.to_csv(test_file, index=False)

    test_prediction['mean'] = np.mean(test_prediction.values, axis=1)
    test_prediction.to_csv(test_file, index=False)

    print(f'{args.n_fold}-fold f1 mean : {np.mean(validation_f1)} std : {np.std(validation_f1)}')
    