import os
import torch
import numpy as np
from sklearn.metrics import f1_score

class Trainer():
    def __init__(self, args, train_loader, valid_loader, model, optimizer, loss_fn, device, model_path):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.model_path = model_path

    def train_epoch(self):
        loss = []
        f1 = []
        pos_acc = []
        neg_acc = []
        self.model.train()
        for batch in self.train_loader:
            x,y = batch['x'].to(self.device), batch['y'].to(self.device).unsqueeze(-1)

            prediction = self.model(x)
            train_loss = self.loss_fn(prediction, y) #compute loss for train set, graph(only with train node)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            prediction = torch.sigmoid(prediction).detach().cpu().numpy()
            prediction[prediction <= 0.5] = 0
            prediction[prediction > 0.5] = 1

            train_f1_score = f1_score(batch['y'], prediction)
            loss.append(train_loss.item())
            f1.append(train_f1_score)
            
            pos_acc.append(sum(prediction[batch['y']==1] == 1) / sum(batch['y']==1))
            neg_acc.append(sum(prediction[batch['y']==0] == 0) / sum(batch['y']==0))

        return sum(loss)/len(loss), sum(f1)/len(f1), float(sum(pos_acc)/len(pos_acc)), float(sum(neg_acc)/len(neg_acc))

    def valid_epoch(self):
        loss = []
        f1 = []
        pos_acc = []
        neg_acc = []
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                x,y = batch['x'].to(self.device), batch['y'].to(self.device).unsqueeze(-1)

                prediction = self.model(x)
                valid_loss = self.loss_fn(prediction, y)
                
                prediction = torch.sigmoid(prediction).detach().cpu().numpy()
                prediction[prediction <= 0.5] = 0
                prediction[prediction > 0.5] = 1

                valid_f1_score = f1_score(batch['y'], prediction)
                loss.append(valid_loss.item())
                f1.append(valid_f1_score)

                pos_acc.append(sum(prediction[batch['y']==1] == 1) / sum(batch['y']==1))
                neg_acc.append(sum(prediction[batch['y']==0] == 0) / sum(batch['y']==0))

        return sum(loss)/len(loss), sum(f1)/len(f1), float(sum(pos_acc)/len(pos_acc)), float(sum(neg_acc)/len(neg_acc))
                    
    def train(self):
        self.best_epoch = None
        best_f1 = 0
        for epoch in range(1,self.args.epochs+1):
            train_loss, train_f1, t_pos_acc, t_neg_acc = self.train_epoch()
            valid_loss, valid_f1, v_pos_acc, v_neg_acc = self.valid_epoch()
            torch.save(self.model.state_dict(), os.path.join(self.model_path, f'{epoch}.pt'))

            print(f"{str(epoch).zfill(3)}-epoch: t_loss {train_loss:.4f} t_f1 {train_f1:.4f} t_pa {t_pos_acc:.4f} t_na {t_neg_acc:.4f} v_loss {valid_loss:.4f} v_f1 {valid_f1:.4f} v_pa {v_pos_acc:.4f} v_na {v_neg_acc:.4f}")
            
            if best_f1 < valid_f1:
                self.best_epoch = epoch
                best_f1 = valid_f1
        
        print(f'valid best_epoch : {self.best_epoch} f1 : {best_f1}')
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, f'{self.best_epoch}.pt')))

        return best_f1
            
    def test(self, loader):
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, f'{self.best_epoch}.pt')))

        test_prediction = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(self.device)
                prediction = self.model(x)

                prediction = torch.sigmoid(prediction).detach().cpu().numpy()
                test_prediction.extend(list(prediction.squeeze(-1)))
        
        return test_prediction
