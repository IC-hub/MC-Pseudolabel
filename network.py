import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import *

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class GPUMLPClassifier:
    def __init__(self, seed, hidden_size=[100], lr=0.001, batch_size=200, max_iter=1000, eval_steps=20, early_stop_steps=3, device='cuda'):
        self.seed = seed
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eval_steps = eval_steps
        self.early_stop_steps = early_stop_steps
        self.device = device

    
    def fit(self, X, y):
        self.clf =  MLP2(
            input_dim = X.shape[1], 
            hidden_size = self.hidden_size, 
            output_features = len(np.unique(y))
        ).to(self.device)
        optimizer = torch.optim.Adam(self.clf.model().parameters(), lr=self.lr)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.seed)
        dataset_train = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
        dataset_val = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
        loader_train = DataLoader(
            dataset_train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=1, 
            pin_memory=True,
            drop_last=False
        )
        loader_val = DataLoader(
            dataset_val, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=1, 
            pin_memory=True,
            drop_last=False
        )
        
        # start training
        iterator = iter(cycle(loader_train))
        min_valid_loss = float("inf")
        best_step = self.max_iter - 1
        for i in range(self.max_iter):
            self.clf.train()
            input_batch, label_batch = next(iterator)
            input_batch = input_batch.to(self.device)
            label_batch = label_batch.to(self.device)
            
            predict, _, _ = self.clf(input_batch)
            predict = predict.squeeze()
            loss = ce_loss_vanilla()(predict, label_batch, None)
            loss.backward()
            optimizer.step()
            self.clf.zero_grad()

            if self.eval_steps > 0 and (i+1) % self.eval_steps == 0:
                train_loss = loss.detach()
                self.clf.eval()
                sample = 0
                loss = 0
                with torch.no_grad():
                    for bundle_batch in loader_val:
                        input_batch, label_batch = bundle_batch[0], bundle_batch[1]
                        batch_size = input_batch.shape[0]
                        input_batch = input_batch.to(self.device)
                        label_batch = label_batch.to(self.device)
                        sample += batch_size
                        predict, _, _ = self.clf(input_batch) # [batch_size]
                        predict = predict.squeeze()
                        loss += ce_loss_vanilla()(predict, label_batch, None) * batch_size           
                val_loss = loss / sample
                print ("DensityRatioClassifier--train-loss-{}-val-loss-{}".format(train_loss.cpu().numpy(), val_loss.cpu().numpy()))
                # record best step
                if val_loss < min_valid_loss:
                    best_step = i
                    min_valid_loss = val_loss
                if (i - best_step) / self.eval_steps >= self.early_stop_steps:
                    break
        return self
    
    def predict_proba(self, X):
        dataset_test= TensorDataset(torch.from_numpy(X).float())
        loader_test = DataLoader(
            dataset_test, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=1, 
            pin_memory=True,
            drop_last=False
        )
        # Inference
        self.clf.eval()
        prob_list = []
        with torch.no_grad():
            for bundle in loader_test:
                input_batch = bundle[0]
                input_batch = input_batch.to(self.device)
                predict, _, _ = self.clf(input_batch) # [batch_size]
                predict = predict.squeeze()
                prob = torch.softmax(predict, dim=1)
                prob_list.append(prob)  
        prob = torch.cat(prob_list, dim=0).cpu().numpy()
        return prob 
