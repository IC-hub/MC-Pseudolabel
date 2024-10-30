import numpy as np
import torch
from tqdm import tqdm
from model import *
from metric import compute_metrics
from register import Register
from dataset import *


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

global trainer_register
trainer_register = Register('trainer_register')


@trainer_register.register
class ERM(object):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, ckpt_dir, tb_writer=None, reset_model=True, **kwargs):
        self._device = device
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._loss_fn = loss_fn
        self._regularizer = regularizer
        self._tb_writer = tb_writer
        self._reg_lambda = kwargs['reg_lambda']
        
        if reset_model:
            reset_parameters(self._model, kwargs['model_init'])


    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        iterator = iter(cycle(self._dataset.training_loader))
        min_valid_loss = float("inf")
        best_step = start_step + num_training_updates - 1
        for i in tqdm(range(num_training_updates), desc='Training'):
            self._model.train()
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            label_batch = label_batch.to(self._device).squeeze()
            env_batch = env_batch.to(self._device)
            
            predict, phi, _ = self._model(input_batch)
            predict = predict.squeeze()
            loss = self._loss_fn(predict, label_batch, env_batch, use_weight=False, reduction='mean')
            if self._regularizer is not None:
                regulation = self._regularizer(predict, phi, label_batch, env_batch, network=self._model.head(), risk=self._loss_fn)     
            else:
                regulation = 0
            regulation_adjust = self._reg_lambda * regulation
            total_loss = loss + regulation_adjust
            total_loss.backward()
            self._optimizer.step()
            self._model.zero_grad()

            if eval_steps > 0 and (i+1) % eval_steps == 0:
                if kwargs['save_model']:
                    save(self._model, self._ckpt_dir, i+start_step)
                val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar('val/loss', val_loss, global_step=i+start_step)
                    for key, val in metric_dict:
                        self._tb_writer.add_scalar(f'val/{key}', val, global_step=i+start_step)
                print (f'train loss:{loss} val loss:{val_loss}')
                print (metric_dict)

                # record best step
                if kwargs['reload_best'] and val_loss < min_valid_loss:
                    best_step = i + start_step
                    min_valid_loss = val_loss

            if logging_steps > 0 and (i+1) % logging_steps == 0:
                print (f'train loss:{loss}') 
                print (f'regularization:{regulation}')
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar("train/loss", loss, global_step=i+start_step)
                    self._tb_writer.add_scalar("train/regularization", regulation, global_step=i+start_step)
            
        return best_step

    

    def evaluate(self, dataloader, metrics=[], loss_reduction=True, return_loss=True, return_predict=False):
        '''
            return: predict_matrix, label_array     if return_predict=True return_loss=False
                    loss, predict_matrix, label_array     if return_predict=True return_loss=True
        '''
        self._model.eval()
        sample = 0
        loss_list = []
        loss = 0
        predict_list = []
        label_list = []
        with torch.no_grad():
            for bundle_batch in tqdm(dataloader, desc='Evaluating'):
                input_batch, label_batch = bundle_batch[0], bundle_batch[1]
                batch_size = input_batch.shape[0]
                input_batch = input_batch.to(self._device)
                label_batch = label_batch.to(self._device).squeeze()
                sample += batch_size
                predict, phi, _ = self._model(input_batch) # [batch_size]
                predict = predict.squeeze()
                
                predict_list.append(predict)
                label_list.append(label_batch)
                
                if return_loss:
                    env_batch = bundle_batch[2]
                    env_batch = env_batch.to(self._device)
                    # calculate loss
                    if loss_reduction:
                        loss += self._loss_fn(predict, label_batch, env_batch) * batch_size
                    else:
                        loss_list.append(self._loss_fn(predict, label_batch, env_batch, reduction='none'))
                    
            predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
            label_array = torch.cat(label_list, dim=0).cpu().numpy()

            if return_loss:
                if loss_reduction:
                    loss /= sample
                else:
                    loss = torch.cat(loss_list)
            
            if return_predict:
                if return_loss:
                    return loss, predict_matrix, label_array
                else:
                    return predict_matrix, label_array
            
        
        if len(metrics)>0:
            metric_dict = compute_metrics(predict_matrix, label_array, metrics=metrics)
            if return_loss:
                return loss, metric_dict
            else:
                return metric_dict
        else:  
            return loss

from hclass import *

@trainer_register.register
class MCPseudolabel(object):
    def __init__(self, device, model, optimizer, dataset, loss_fn, ckpt_dir, **kwargs):
        self._device = device
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._dataset = dataset
        self._loss_fn = loss_fn
        self._ckpt_dir = ckpt_dir
        self.erm = ERM(self._device, self._model, self._optimizer, self._dataset, self._loss_fn, regularizer=None, ckpt_dir=self._ckpt_dir, tb_writer=None, **kwargs)

        if kwargs['mc_hclass'] == 'LogitDensityRatioHClass':
            self._hclass = LogitDensityRatioHClass(kwargs['seed'])
            X, y, e, _ = self._dataset.training_dataset.tensors
            X = X.numpy()
            y = y.numpy()
            e = e.numpy()

            self._hclass.fit(X, y, e)
            print ('Density Ratio Function:', self._hclass.function())
            
            if kwargs['verbose']:
                p = self._hclass.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
                import matplotlib.pyplot as plt
                plt.hist(p[:, 0], bins=30, edgecolor='black') 
                plt.title('Probability Distribution for e=0')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
        elif kwargs['mc_hclass'] == 'RidgeDensityRatioHClass':
            self._hclass = RidgeDensityRatioHClass(kwargs['seed'])
            X, y, e, _ = self._dataset.training_dataset.tensors
            X = X.numpy()
            y = y.numpy()
            e = e.numpy()

            self._hclass.fit(X, y, e)
            print ('Density Ratio Function:', self._hclass.function())
            
            if kwargs['verbose']:
                p = self._hclass.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
                import matplotlib.pyplot as plt
                plt.hist(p[:, 0], bins=30, edgecolor='black') 
                plt.title('Probability Distribution for e=0')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
        elif kwargs['mc_hclass'] == 'NeuralDensityRatioHClass':
            self._hclass = NeuralDensityRatioHClass(kwargs['seed'])
            X, y, e, _ = self._dataset.training_dataset.tensors
            X = X.numpy()
            y = y.numpy()
            e = e.numpy()

            if kwargs['domain_clf_lr'] is None:
                self._hclass.fit(X, y, e)
            else:
                self._hclass.fit(X, y, e, lr=kwargs['domain_clf_lr'])
            
            if kwargs['verbose']:
                p = self._hclass.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
                import matplotlib.pyplot as plt
                plt.hist(p[:, 0], bins=30, edgecolor='black') 
                plt.title('Probability Distribution for e=0')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
        elif kwargs['mc_hclass'] == 'GBDTDensityRatioHClass':
            self._hclass = GBDTDensityRatioHClass(kwargs['seed'])
            X, y, e, _ = self._dataset.training_dataset.tensors
            X = X.numpy()
            y = y.numpy()
            e = e.numpy()

            self._hclass.fit(X, y, e)
            
            if kwargs['verbose']:
                p = self._hclass.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
                import matplotlib.pyplot as plt
                plt.hist(p[:, 0], bins=30, edgecolor='black') 
                plt.title('Probability Distribution for e=0')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
        elif kwargs['mc_hclass'] == 'NeuralGPUDensityRatioHClass':
            self._hclass = NeuralGPUDensityRatioHClass(kwargs['seed'])
            X, y, e, _ = self._dataset.training_dataset.tensors
            X = X.numpy()
            y = y.numpy()
            e = e.numpy()
            self._hclass.fit(X, y, e)
            
            if kwargs['verbose']:
                p = self._hclass.clf.predict_proba(np.concatenate((X,y.reshape(-1,1)), axis=1))
                import matplotlib.pyplot as plt
                plt.hist(p[:, 0], bins=30, edgecolor='black') 
                plt.title('Probability Distribution for e=0')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
        elif kwargs['mc_hclass'] == 'HardSampleHClass':
            self._hclass = HardSampleHClass(kwargs['seed'])
            X, y, e, _ = self._dataset.training_dataset.tensors
            X = X.numpy()
            y = y.numpy()
            e = e.numpy()

            self._hclass.fit(X, y, e)
        elif kwargs['mc_hclass'] == 'PovertyDensityRatioHClass':
            self._hclass = PovertyDensityRatioHClass(self._device)
            self._hclass.fit(self._dataset)
            
            if kwargs['verbose']:
                p = self._hclass.predict_matrix
                import matplotlib.pyplot as plt
                plt.hist(p[:, 0], bins=30, edgecolor='black') 
                plt.title('Probability Distribution for e=0')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
        else:
            raise NotImplementedError

    
    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, mc_pretrain=True, **kwargs):
        if mc_pretrain:
            print("First training with ERM.") 
            self.erm.train(num_training_updates, logging_steps, eval_steps, metrics, start_step, **kwargs)
        
        print("MCPseudolabeling.")
        for t in tqdm(range(kwargs['num_mc_updates']), desc='MCPseudolabeling'):
            predict_matrix, label_array = self.erm.evaluate(self._dataset.training_loader_sequential, return_loss=False, return_predict=True)
            if t==0:
                if kwargs['mc_round_interval'] == 0:
                    m = m_estimate_percentile(predict_matrix, n_sample=100, percentile=90)
                else:
                    m = kwargs['mc_round_interval']
                m = max(m, 10)
                print ("m:", m)
            rounded_f = round_function(predict_matrix, m)
            
            if kwargs['mc_hclass'] == 'PovertyDensityRatioHClass':
                X = None
            else:
                X, _, _, _ = self._dataset.training_dataset.tensors
                X = X.numpy()

            y_tilde = level_regression(self._hclass, X, label_array, rounded_f)
            
            err = ((rounded_f-label_array)*(rounded_f-label_array)).mean()
            err_tilde = ((y_tilde-label_array)*(y_tilde-label_array)).mean()
            print ('err:', err, 'err_tilde', err_tilde, 'gap', err-err_tilde)

            if kwargs['mc_hclass'] == 'PovertyDensityRatioHClass':
                y_tilde_tensor = torch.from_numpy(y_tilde).float()
                iterator = iter(cycle(self._dataset.training_loader))
                self._model.train()
                for i in range(kwargs['mc_finetune_steps']):     
                    input_batch, _, _, ids = next(iterator)
                    input_batch = input_batch.to(self._device)
                    label_batch = y_tilde_tensor[ids].to(self._device).squeeze()
                    
                    predict, _ , _ = self._model(input_batch)
                    predict = predict.squeeze()
                    # only vannila loss allowed
                    loss = self._loss_fn(predict, label_batch, None, use_weight=False, reduction='mean')
                    loss.backward()
                    self._optimizer.step()
                    self._model.zero_grad()
            else:
                Dt = TensorDataset(self._dataset.training_dataset.tensors[0], torch.from_numpy(y_tilde).float())
                Dt_loader = DataLoader(
                    dataset=Dt,
                    batch_size=kwargs['batch_size'],
                    shuffle=True,
                    num_workers=kwargs['num_workers'], 
                    pin_memory=True,
                    drop_last=False
                )
                iterator = iter(cycle(Dt_loader))
                self._model.train()
                for i in range(kwargs['mc_finetune_steps']):                   
                    input_batch, label_batch = next(iterator)
                    input_batch = input_batch.to(self._device)
                    label_batch = label_batch.to(self._device).squeeze()
                    
                    predict, _ , _ = self._model(input_batch)
                    predict = predict.squeeze()
                    # only vannila loss allowed
                    loss = self._loss_fn(predict, label_batch, None, use_weight=False, reduction='mean')
                    loss.backward()
                    self._optimizer.step()
                    self._model.zero_grad()
        
        val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
        print (f'val loss:{val_loss} metric:{metric_dict}')
        metric_dict['err_inv'] = 2*val_loss.detach().cpu().numpy() - err_tilde
        print (metric_dict)
        
        return metric_dict
    
    def evaluate(self, dataloader, metrics=[], loss_reduction=True, return_loss=True, return_predict=False):
        return self.erm.evaluate(dataloader, metrics, loss_reduction=loss_reduction, return_loss=return_loss, return_predict=return_predict)    
    
