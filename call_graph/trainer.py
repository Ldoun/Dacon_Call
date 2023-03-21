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
            train_out = self.model(train_tensor_x[train_idx], graph[:, train_index])
            train_loss = self.loss_fn(train_out, train_tensor_y[train_idx]) #compute loss for train set, graph(only with train node)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            prediction = train_out.detach().cpu().numpy()
            prediction[prediction <= 0.5] = 0
            prediction[prediction > 0.5] = 1
            train_f1_score = f1_score(train_y.loc[train_idx], prediction)

        return train_loss.item(), train_f1_score

    def valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.valid_loader:
                valid_out = model(train_tensor_x, graph)
                valid_loss = loss_fn(valid_out[valid_idx], train_tensor_y[valid_idx]) #compute loss for valid set, graph(with train&valid node)
                
                prediction = valid_out.detach().cpu().numpy()
                prediction[prediction <= 0.5] = 0
                prediction[prediction > 0.5] = 1
                valid_f1_score = f1_score(train_y.loc[valid_idx], prediction[valid_idx])

        return train_loss.item(), train_f1_score
                    
    def train(self):
        for epoch in range(1,self.args.epochs+1):
            train_loss, train_f1 = self.train_epoch()
            valid_loss, valid_f1 = self.valid_epoch()
            
            print(f"{epoch}-epoch: t_loss {train_loss.item()} t_f1 {train_f1} v_loss {valid_loss.item()} v_f1 {valid_f1}")
