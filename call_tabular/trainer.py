import torch
from sklearn.metrics import f1_score


class Trainer():
    def __init__(self, args, train_loader, valid_loader, model, optimizer, loss_fn,):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def train_epoch(self):
        self.model.train()
        for batch in self.train_loader:
            prediction = self.model(batch['x'])
            train_loss = self.loss_fn(prediction, batch['y']) #compute loss for train set, graph(only with train node)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            prediction = prediction.detach().cpu().numpy()
            prediction[prediction <= 0.5] = 0
            prediction[prediction > 0.5] = 1
            train_f1_score = f1_score(batch['y'].detach().cpu().numpy(), prediction)

        return train_loss.item(), train_f1_score

    def valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                prediction = self.model(batch['x'])
                valid_loss = self.loss_fn(prediction, batch['y'])
                
                prediction = prediction.detach().cpu().numpy()
                prediction[prediction <= 0.5] = 0
                prediction[prediction > 0.5] = 1
                valid_f1_score = f1_score(batch['y'].detach().cpu().numpy(), prediction)

        return valid_loss.item(), valid_f1_score
                    
    def train(self):
        for epoch in range(1,self.args.epochs+1):
            train_loss, train_f1 = self.train_epoch()
            valid_loss, valid_f1 = self.valid_epoch()
            
            print(f"{epoch}-epoch: t_loss {train_loss} t_f1 {train_f1} v_loss {valid_loss} v_f1 {valid_f1}")
