import numpy as np
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression, LinearRegression
from model import *
from poverty import *
import torch
from copy import deepcopy
from metric import compute_metrics

class LogitDensityRatioHClass:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, e):
        self.clf = LogisticRegression(
            random_state=self.seed, 
            multi_class='multinomial', 
            max_iter=1000, 
            penalty='none').fit(np.concatenate((X,y.reshape(-1,1)), axis=1), e)
        return self
    
    def regression(self, X, y):
        hvalue = self.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
        y_hat = LinearRegression().fit(hvalue, y).predict(hvalue)
        return y_hat
    
    def function(self):
        return self.clf.coef_, self.clf.intercept_

class RidgeDensityRatioHClass:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, e):
        self.clf = LogisticRegression(
            random_state=self.seed, 
            multi_class='multinomial', 
            max_iter=1000, 
            penalty='l2').fit(np.concatenate((X,y.reshape(-1,1)), axis=1), e)
        return self
    
    def regression(self, X, y):
        hvalue = self.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
        y_hat = LinearRegression().fit(hvalue, y).predict(hvalue)
        return y_hat
    
    def function(self):
        return self.clf.coef_, self.clf.intercept_

from sklearn.neural_network import MLPClassifier

class NeuralDensityRatioHClass:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, e, lr=0.001):
        self.clf = MLPClassifier(
            random_state=self.seed, 
            max_iter=1000,
            verbose=True,
            learning_rate_init=lr).fit(np.concatenate((X,y.reshape(-1,1)), axis=1), e)
        
        return self
    
    def regression(self, X, y):
        hvalue = self.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
        y_hat = LinearRegression().fit(hvalue, y).predict(hvalue)
        return y_hat
    

    
from sklearn.ensemble import GradientBoostingClassifier

class GBDTDensityRatioHClass:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, e):
        self.clf = GradientBoostingClassifier(
            random_state=self.seed, 
            verbose=True).fit(np.concatenate((X,y.reshape(-1,1)), axis=1), e)
        
        return self
    
    def predict(self, X, y):
        hvalue = self.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
        return hvalue

from network import *



class NeuralGPUDensityRatioHClass:
    def __init__(self, seed, hidden_size=[64,16], lr=0.005, batch_size=1024, max_iter=1000, eval_steps=20, early_stop_steps=5, device='cuda'):
        self.seed = seed
        self.hidden_size = hidden_size
        self.lr = lr
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.eval_steps =eval_steps
        self.early_stop_steps = early_stop_steps
        self.device = device

    def fit(self, X, y, e):
        self.clf = GPUMLPClassifier(
            seed = self.seed,
            hidden_size = self.hidden_size,
            lr = self.lr,
            batch_size = self.batch_size,
            max_iter = self.max_iter,
            eval_steps = self.eval_steps,
            early_stop_steps = self.early_stop_steps,
            device=self.device
        ).fit(np.concatenate((X,y.reshape(-1,1)), axis=1), e)
        return self
    
    def predict(self, X, y):
        hvalue = self.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
        return hvalue

from sklearn.linear_model import Ridge

    
class HardSampleHClass:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y, e):
        self.model = Ridge().fit(X, y)
        return self
    
    def regression(self, X, y):
        hvalue = (self.model.predict(X)-y)**2
        y_hat = LinearRegression().fit(hvalue.reshape(-1,1), y).predict(hvalue.reshape(-1,1))
        return y_hat
    
    def predict(self, X, y):
        hvalue = self.model.predict(X)
        return hvalue.reshape(-1,1)

class PovertyDensityRatioHClass:
    def __init__(self, device):
        self.device = device
        self.predict_matrix = None

    def fit(self, _dataset, lr=0.001, max_iter=400, eval_steps=200, early_stop_steps=3): #1600
        dataset = deepcopy(_dataset)
        model = ResnetMS(num_classes=2, input_label=True)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        iterator = iter(cycle(dataset.training_loader))
        min_valid_loss = float("inf")
        best_step = max_iter - 1
        for i in range(max_iter):
            model.train()
            input_batch, target_batch, label_batch, _ = next(iterator)
            input_batch = input_batch.to(self.device)
            target_batch = target_batch.to(self.device)
            label_batch = label_batch.to(self.device).squeeze()
            
            predict, _, _ = model(input_batch, target_batch)
            predict = predict.squeeze()
            loss = ce_loss_vanilla()(predict, label_batch, None)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            if eval_steps > 0 and (i+1) % eval_steps == 0:
                train_loss = loss.detach()
                model.eval()
                sample = 0
                loss = 0
                predict_list = []
                label_list = []
                with torch.no_grad():
                    for bundle_batch in dataset.validation_loader:
                        input_batch, target_batch, label_batch = bundle_batch[0], bundle_batch[1], bundle_batch[2]
                        batch_size = input_batch.shape[0]
                        input_batch = input_batch.to(self.device)
                        target_batch = target_batch.to(self.device)
                        label_batch = label_batch.to(self.device).squeeze()
                        sample += batch_size
                        predict, _, _ = model(input_batch, target_batch) # [batch_size]
                        predict = predict.squeeze()
                        loss += ce_loss_vanilla()(predict, label_batch, None) * batch_size    
                        predict_list.append(predict)     
                        label_list.append(label_batch)   
                val_loss = loss / sample
                predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
                label_matrix = torch.cat(label_list, dim=0).cpu().numpy()
                accuracy = compute_metrics(predict_matrix, label_matrix, metrics=['AccuracyMulticlass'])['AccuracyMulticlass']
                print ("DensityRatioClassifier--train-loss-{}-val-loss-{}-val-accuracy-{}".format(train_loss.cpu().numpy(), val_loss.cpu().numpy(), accuracy))
                # record best step
                if val_loss < min_valid_loss:
                    best_step = i
                    min_valid_loss = val_loss
                if (i - best_step) / eval_steps >= early_stop_steps:
                    break
        model.eval()
        predict_list = []
        with torch.no_grad():
            for bundle_batch in dataset.training_loader_sequential:
                input_batch, target_batch = bundle_batch[0], bundle_batch[1]
                input_batch = input_batch.to(self.device)
                target_batch = target_batch.to(self.device)
                predict, _, _ = model(input_batch, target_batch) # [batch_size]
                predict = predict.squeeze()
                predict = torch.softmax(predict, dim=1)
                predict_list.append(predict)         
            self.predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
        return self
    
    def predict(self, X=None, y=None):
        return self.predict_matrix


# Define apply_regression at the top level
def apply_regression(args):
    h_class, X, y, f, value = args

    subset_mask = (f == value)
    X_subset = X[subset_mask]
    y_subset = y[subset_mask]
    res = h_class.regression(X_subset, y_subset)

    return res

def densityRatioHClassRegression(hvalue, y):
    y_hat = LinearRegression().fit(hvalue, y).predict(hvalue)
    return y_hat

def apply_regression_with_hvalue(args):
    hvalue, y, f, value = args

    subset_mask = (f == value)
    hvalue_subset = hvalue[subset_mask]
    y_subset = y[subset_mask]
    res = densityRatioHClassRegression(hvalue_subset, y_subset)

    return res

def level_regression(h_class, X, y, f, n_proc=20):
    unique_values = np.unique(f)

    if h_class.__class__.__name__ == 'NeuralGPUDensityRatioHClass' or \
        h_class.__class__.__name__ == 'GBDTDensityRatioHClass' or \
        h_class.__class__.__name__ == 'HardSampleHClass' or \
        h_class.__class__.__name__ == 'PovertyDensityRatioHClass' :
        hvalue = h_class.predict(X, y)
        args = [(hvalue, y, f, value) for value in unique_values]
        with Pool(processes=n_proc) as pool:
            results = pool.map(apply_regression_with_hvalue, args)
    else:
        # Create arguments for apply_regression
        args = [(h_class, X, y, f, value) for value in unique_values]
        with Pool(processes=n_proc) as pool:
            results = pool.map(apply_regression, args)
        

    y_hat = np.zeros_like(f)
    for value, result in zip(unique_values, results):
        if result is not None:   
            y_hat[f == value] = result
        else:
            raise NotImplementedError

    return y_hat

def round_function(f, m, a=None, b=None):
    if a is None or b is None:
        a = np.min(f)
        b = np.max(f)
    
    # Discretize the values into m intervals
    # We create m equally spaced intervals between a and b, and use np.digitize to map values to these intervals
    intervals = np.linspace(a, b, m+1)
    discretized_f = np.digitize(f, intervals, right=True)

    # Map the discretized values back to interval midpoints for rounding
    interval_midpoints = (intervals[:-1] + intervals[1:]) / 2
    rounded_f = np.array([interval_midpoints[i-1] if i > 0 else a for i in discretized_f])

    return rounded_f

def round_function_equal_interval(arr, bin_size=30):
    
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]

  
    bin_means = np.array([np.mean(sorted_arr[i:i+bin_size]) for i in range(0, len(sorted_arr), bin_size)])

   
    bin_indices = np.arange(len(arr)) // bin_size
    discretized = bin_means[bin_indices]

    
    original_order = np.argsort(sorted_indices)
    discretized_in_original_order = discretized[original_order]

    return discretized_in_original_order

def m_tol(f, tol, a=None, b=None):
    if a is None or b is None:
        a = np.min(f)
        b = np.max(f)

    m = int((b-a)*(b-a)/tol)
    return m

def m_estimate(f, n_sample):
    m = int(len(f)/n_sample)
    return m

def m_estimate_percentile(f, n_sample, percentile):
    ''' $percentile percent of the data in f is covered by a bin with at least $n_sample samples.
    '''
    m = int(len(f) / n_sample)
    rounded_f = round_function(f, m)
    
    def coverage(rounded_f):
        sum = 0
        for value in np.unique(rounded_f):
            if len(rounded_f[rounded_f==value]) >= n_sample:
                sum += len(rounded_f[rounded_f==value])
        return sum/len(f)*100

    while (coverage(rounded_f) < percentile):
        m -= 5
        rounded_f = round_function(f, m)
    return m

