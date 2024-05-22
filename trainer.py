import os
import numpy as np
# from sklearn.neighbors import kneighbors_graph
import torch
import torch.nn as nn
# import pynndescent
from tqdm import tqdm
from model import *
from metric import compute_metrics
from register import Register
from dataset import *
from copy import deepcopy



def cycle(iterable):
    while True:
        for x in iterable:
            yield x

global trainer_register
trainer_register = Register('trainer_register')

def evaluate_env(dataloader, model, device, metrics=[]):
        '''
            return: predict_matrix, label_array     if return_predict=True return_loss=False
                    loss, predict_matrix, label_array     if return_predict=True return_loss=True
        '''
        model.eval()
        predict_list = []
        label_list = []
        env_list = []
        with torch.no_grad():
            for bundle_batch in tqdm(dataloader, desc='Evaluating'):
                input_batch, label_batch, env_batch = bundle_batch[0], bundle_batch[1], bundle_batch[2]
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device).squeeze()
                predict, _, _ = model(input_batch) # [batch_size]
                predict = predict.squeeze()
                predict_list.append(predict)
                label_list.append(label_batch)
                env_list.append(env_batch)
                    
            predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
            label_array = torch.cat(label_list, dim=0).cpu().numpy()        
            env_array = torch.cat(env_list, dim=0).cpu().numpy()    
        
        assert (len(metrics)>0)
        metric_dict_list = []
        for env in np.unique(env_array):
            metric_dict = compute_metrics(predict_matrix[env_array==env], label_array[env_array==env], metrics=metrics)
            metric_dict_list.append(metric_dict)
        
        average_metric_dict = compute_metrics(predict_matrix, label_array, metrics=metrics)

        return average_metric_dict, metric_dict_list



@trainer_register.register
class ERM(object):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, ckpt_dir, tb_writer=None, reset_model=True, **kwargs):
        self._device = device
        self._model = model.to(self._device)
        # self._head = head.to(self._device)
        self._optimizer = optimizer
        # self._optimizer_head = optimizer_head
        self._dataset = dataset
        self._ckpt_dir = ckpt_dir
        self._loss_fn = loss_fn
        self._regularizer = regularizer
        self._tb_writer = tb_writer
        self._reg_lambda = kwargs['reg_lambda']
        # self._model.restore(ckpt_dir)
        # self._weight = torch.ones(len(dataset.training_dataset)).to(device)
        
        if reset_model:
            reset_parameters(self._model, kwargs['model_init'])
        
        # reset_parameters(self._model.head(), 'constant')
    
    # def update_weight(self, weight):
    #     self._weight = weight.to(self._device)


    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        iterator = iter(cycle(self._dataset.training_loader))
        min_valid_loss = float("inf")
        best_step = start_step + num_training_updates - 1
        for i in tqdm(range(num_training_updates), desc='Training'):
            self._model.train()
            # print (next(iterator))
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

                # loss_inspect = self._loss_fn(predict, label_batch, env_batch, use_weight=False, reduction='none')
                # for env in env_batch.unique():
                #     print (f'{env}:', loss_inspect[env_batch==env].mean().detach().cpu().numpy())
                # if kwargs['verbose']:
                #     print ('params:', next(self._model.model().parameters()).data)            
                    

                # print_loss, _ = self.evaluate(self._dataset.validation_loader, metrics, loss_reduction=False)
                # print ('val', print_loss[:15893].mean(), print_loss[15893:].mean())

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
        # correct_number = 0
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
class MCDeboost(object):
    def __init__(self, device, model, optimizer, dataset, loss_fn, ckpt_dir, **kwargs):
        # self.erm = ERM(device, model, optimizer, dataset, loss_fn, regularizer=None, ckpt_dir=ckpt_dir, tb_writer=None, **kwargs)
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

            # from sklearn.decomposition import PCA
            # X = PCA(n_components=10).fit_transform(X)

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

            # from sklearn.decomposition import PCA
            # X = PCA(n_components=10).fit_transform(X)

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

            # from sklearn.decomposition import PCA
            # X = PCA(n_components=10).fit_transform(X)
            # self.X = X

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

            # from sklearn.decomposition import PCA
            # X = PCA(n_components=10).fit_transform(X)

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
        
        # TODO remove
        # metric_dict, metric_dict_list = evaluate_env(self._dataset.test_loader, self._model, self._device, metrics=metrics)
        # print (metric_dict)
        # print(metric_dict_list)
        
        print("MCDeboosting.")
        # y_tilde_list = []
        # predict_list = []
        for t in tqdm(range(kwargs['num_mc_updates']), desc='MCDeboosting'):
            # if kwargs['verbose']:
            #     print ('model params:', next(self._model.model().parameters()).data)    
            predict_matrix, label_array = self.erm.evaluate(self._dataset.training_loader_sequential, return_loss=False, return_predict=True)
            # predict_list.append(predict_matrix)
            if t==0:
                # err0 = ((predict_matrix-label_array)*(predict_matrix-label_array)).mean()
                if kwargs['mc_round_interval'] == 0:
                    m = m_estimate_percentile(predict_matrix, n_sample=100, percentile=90)
                else:
                    m = kwargs['mc_round_interval']
                m = max(m, 10)
                print ("m:", m)
            rounded_f = round_function(predict_matrix, m)
            # rounded_f = round_function_equal_interval(predict_matrix, 30)
            
            if kwargs['mc_hclass'] == 'PovertyDensityRatioHClass':
                X = None
            else:
                X, _, _, _ = self._dataset.training_dataset.tensors
                X = X.numpy()

            # X = self.X

            y_tilde = level_regression(self._hclass, X, label_array, rounded_f)
            # y_tilde_list.append(y_tilde)
            
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

            # TODO: recover
            # val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
            # print (f'val loss:{val_loss} metric:{metric_dict}')
            
            # TODO: remove
            # metric_dict, metric_dict_list = evaluate_env(self._dataset.test_loader, self._model, self._device, metrics=metrics)
            # print (metric_dict)
            # print(metric_dict_list)

            # metric_dict = self.evaluate(self._dataset.test_loader['test_0'], ['RMSE'], return_loss=False)
            # print (0,metric_dict)
            # metric_dict = self.evaluate(self._dataset.test_loader['test_1'], ['RMSE'], return_loss=False)
            # print (1,metric_dict)
            
            # metric_dict = self.evaluate(self._dataset.test_loader['test'], ['RMSE'], return_loss=False)
            
            # TODO: recover
            # metric_dict['err_inv'] = 2*val_loss.detach().cpu().numpy() - err_tilde
            # print (metric_dict)
        
        val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
        print (f'val loss:{val_loss} metric:{metric_dict}')
        metric_dict['err_inv'] = 2*val_loss.detach().cpu().numpy() - err_tilde
        print (metric_dict)
        
        return metric_dict

        
        # y_tilde_list = [label_array] + y_tilde_list
        # y_tilde = np.stack(y_tilde_list, axis=1)
        # f = np.stack(predict_list, axis=1)
        # np.save('toy_slope/X.npy', X)
        # np.save('toy_slope/y_tilde.npy', y_tilde)
        # np.save('toy_slope/f.npy', f)


            
            
    def evaluate(self, dataloader, metrics=[], loss_reduction=True, return_loss=True, return_predict=False):
        return self.erm.evaluate(dataloader, metrics, loss_reduction=loss_reduction, return_loss=return_loss, return_predict=return_predict)    
    
@trainer_register.register
class GroupDRO(object):
    def __init__(self, device, model, optimizer, dataset, loss_fn, ckpt_dir, **kwargs):
        if type(dataset).__name__ == 'PovertyDataLoader':
            n_env = dataset.training_dataset.num_envs
        else:
            _, _, e, _ = dataset.training_dataset.tensors
            n_env = len(torch.unique(e))
        print("environment number:", n_env)
        groupdro_loss = groupDRO(loss_fn, device, n_env=n_env, eta=kwargs['reg_lambda'])
        self.erm = ERM(device, model, optimizer, dataset, groupdro_loss, regularizer=None, ckpt_dir=ckpt_dir, **kwargs)
    
    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        return self.erm.train(num_training_updates, logging_steps, eval_steps, metrics, start_step, **kwargs)
    
    def evaluate(self, dataloader, metrics=[], loss_reduction=True, return_loss=True, return_predict=False):
        return self.erm.evaluate(dataloader, metrics, loss_reduction=loss_reduction, return_loss=return_loss, return_predict=return_predict)

@trainer_register.register
class X2DRO(object):
    def __init__(self, device, model, optimizer, dataset, loss_fn, ckpt_dir, **kwargs):
        x2droloss = x2dro_loss(loss_fn, eta=kwargs['reg_lambda'])
        self.erm = ERM(device, model, optimizer, dataset, x2droloss, regularizer=None, ckpt_dir=ckpt_dir, **kwargs)
    
    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        return self.erm.train(num_training_updates, logging_steps, eval_steps, metrics, start_step, **kwargs)
    
    def evaluate(self, dataloader, metrics=[], loss_reduction=True, return_loss=True, return_predict=False):
        return self.erm.evaluate(dataloader, metrics, loss_reduction=loss_reduction, return_loss=return_loss, return_predict=return_predict)
    
@trainer_register.register
class TERM(object):
    def __init__(self, device, model, optimizer, dataset, loss_fn, ckpt_dir, **kwargs):
        tloss = tilted_loss(loss_fn, t=kwargs['reg_lambda'])
        self.erm = ERM(device, model, optimizer, dataset, tloss, regularizer=None, ckpt_dir=ckpt_dir, **kwargs)
    
    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        return self.erm.train(num_training_updates, logging_steps, eval_steps, metrics, start_step, **kwargs)
    
    def evaluate(self, dataloader, metrics=[], loss_reduction=True, return_loss=True, return_predict=False):
        return self.erm.evaluate(dataloader, metrics, loss_reduction=loss_reduction, return_loss=return_loss, return_predict=return_predict)
    
    
@trainer_register.register
class CVaR(ERM):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, ckpt_dir, tb_writer=None, reset_model=True, **kwargs):
        super(CVaR, self).__init__(device, model, optimizer, dataset, loss_fn, regularizer, ckpt_dir, tb_writer, reset_model, **kwargs)
        self.alpha = kwargs['alpha_threshold']


    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        iterator = iter(cycle(self._dataset.training_loader))
        min_valid_loss = float("inf")
        best_step = start_step + num_training_updates - 1
        for i in tqdm(range(num_training_updates), desc='Training'):
            self._model.train()
            # print (next(iterator))
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            label_batch = label_batch.to(self._device).squeeze()
            env_batch = env_batch.to(self._device)
            
            predict, _, _ = self._model(input_batch)
            predict = predict.squeeze()
            loss = self._loss_fn(predict, label_batch, env_batch, use_weight=False, reduction='none')
            loss = loss[loss>torch.quantile(loss, self.alpha)].mean()
            loss.backward()
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

                loss_inspect = self._loss_fn(predict, label_batch, env_batch, use_weight=False, reduction='none')
                for env in env_batch.unique():
                    print (f'{env}:', loss_inspect[env_batch==env].mean().detach().cpu().numpy())

                print (metric_dict)

                # record best step
                if kwargs['reload_best'] and val_loss < min_valid_loss:
                    best_step = i + start_step
                    min_valid_loss = val_loss
                

            if logging_steps > 0 and (i+1) % logging_steps == 0:
                print (f'train loss:{loss}') 
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar("train/loss", loss, global_step=i+start_step)
            
        return best_step

@trainer_register.register
class JTT(ERM):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, ckpt_dir, tb_writer=None, reset_model=True, **kwargs):
        super(JTT, self).__init__(device, model, optimizer, dataset, loss_fn, regularizer, ckpt_dir, tb_writer, reset_model, **kwargs)
        self.alpha = kwargs['alpha_threshold']
        self.lambda_up = kwargs['jtt_lambda_up']


    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        X, y, _, _ = self._dataset.training_dataset.tensors
        X = X.numpy()
        y = y.numpy()

        from sklearn.linear_model import Ridge
        model_id = Ridge().fit(X, y)
        err = (model_id.predict(X)-y)**2
        err = torch.from_numpy(err).float().to(self._device)
        sample_weight = torch.ones(len(err)).to(self._device)
        sample_weight[err>torch.quantile(err, self.alpha)] = self.lambda_up

        iterator = iter(cycle(self._dataset.training_loader))
        min_valid_loss = float("inf")
        best_step = start_step + num_training_updates - 1
        for i in tqdm(range(num_training_updates), desc='Training'):
            self._model.train()
            # print (next(iterator))
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            label_batch = label_batch.to(self._device).squeeze()
            env_batch = env_batch.to(self._device)
            ids = ids.to(self._device)
            
            predict, _, _ = self._model(input_batch)
            predict = predict.squeeze()
            loss = self._loss_fn(predict, label_batch, env_batch, use_weight=False, reduction='none')
            loss = (loss * sample_weight[ids]).mean() / sample_weight[ids].mean()
            loss.backward()
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

                loss_inspect = self._loss_fn(predict, label_batch, env_batch, use_weight=False, reduction='none')
                for env in env_batch.unique():
                    print (f'{env}:', loss_inspect[env_batch==env].mean().detach().cpu().numpy())

                print (metric_dict)

                # record best step
                if kwargs['reload_best'] and val_loss < min_valid_loss:
                    best_step = i + start_step
                    min_valid_loss = val_loss
                

            if logging_steps > 0 and (i+1) % logging_steps == 0:
                print (f'train loss:{loss}') 
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar("train/loss", loss, global_step=i+start_step)
            
        return best_step

@trainer_register.register
class JTTGWDRO(object):
    def __init__(self, 
            device, 
            model, 
            optimizer, 
            dataset, 
            loss_fn:Loss, 
            ckpt_dir,
            tb_writer=None,
            **kwargs
    ):
        self.dataset = dataset
        self._tb_writer = tb_writer
        self._model_init = kwargs['model_init']
        self.erm = ERM(device, model, optimizer, dataset, loss_fn, ckpt_dir, tb_writer, **kwargs)
        print ('Constructing knn graph.')

        from sklearn.model_selection import train_test_split


        if kwargs['nndescent']:
            knn = pynndescent.PyNNDescentTransformer(n_neighbors=kwargs['k_neighbor'], random_state=kwargs['seed'], n_jobs=20)
            adj_m = knn.fit_transform(np.array(dataset.training_dataset.tensors[0].cpu()))
            # t1 = dataset.training_dataset.tensors[0].cpu()[:169752]
            # t1 = train_test_split(t1, train_size=0.05)[0]
            # t2 = dataset.training_dataset.tensors[0].cpu()[169752:]
            # t2 = train_test_split(t2, train_size=0.05)[0]
            # t = torch.cat([t1, t2])
            # adj_m = knn.fit_transform(np.array(t))
        else:
            adj_m = kneighbors_graph(np.array(dataset.training_dataset.tensors[0].cpu()), kwargs['k_neighbor'], mode=kwargs['knn_mode'])
        adj_m = (adj_m+adj_m.transpose()) / 2.0
        adj_m = adj_m.tocoo()


        # print ('plotting')

        # cmap = ['#48D1CC'] * len(t1) + ['#8B0000'] * len(t2)


        # G = nx.from_numpy_matrix(adj_m.todense())
        # random_pos = nx.random_layout(G, seed=42)
        # pos = nx.spring_layout(G, pos=random_pos)

        # plt.figure(figsize=(40,40),dpi=100)
        # nx.draw_networkx(G, with_labels=False, pos = pos, node_color=cmap, node_size=5,width=0.1)
        # plt.savefig('covid.png')

        print ('Initializing gradient flow.')
        self.gf = GradientFlow(
                    adj_m, 
                    np.zeros(adj_m.shape[0]), 
                    kwargs['beta'], 
                    kwargs['flow_lr'], 
                    device=device, 
                    interpolation='upwind'
        )
    
    
    def train(self, 
            num_training_updates,
            logging_steps, 
            eval_steps=0, 
            metrics=[],
            **kwargs
    ):
        global_step = 0
        print ('Train model at first time.')
        self.erm.train(num_training_updates, logging_steps, eval_steps, metrics, 0, **kwargs)
        global_step += num_training_updates

        print ('Gradient flow.')
        V = - self.evaluate(self.dataset.training_loader_sequential, metrics=[], loss_reduction=False)
        # print (V.sort())
        self.gf.update_V(V)
        for t in range(kwargs['flow_steps']):
            self.gf.step()
            if kwargs['flow_eval_steps'] > 0 and (t+1) % kwargs['flow_eval_steps'] == 0:
                loss = - (self.gf.rou * V).sum()
                print ('loss:', loss)
                if self._tb_writer is not None:
                    self._tb_writer.add_scalar('train/loss', loss, global_step=global_step)
            global_step += 1
        # self.erm._loss_fn.update_weight(self.gf.rou * len(self.dataset.training_dataset))
        self.erm.update_weight(self.gf.rou * len(self.dataset.training_dataset))

        
        # print (-V[:1625].mean(), -V[1625:].mean())
        # print ('weight', self.gf.rou[:63571].mean(), self.gf.rou[63571:].mean())

        print ('Train model at second time.')    
        reset_parameters(self.erm._model, self._model_init)
        best_step = self.erm.train(num_training_updates, logging_steps, eval_steps, metrics, global_step, **kwargs)

        return best_step
    

    def evaluate(self, dataloader, metrics=[], loss_reduction=True):
        return self.erm.evaluate(dataloader, metrics, loss_reduction=loss_reduction)

@trainer_register.register
class KLDRO(object):
    def __init__(self, 
            device, 
            model, 
            optimizer, 
            dataset, 
            loss_fn:Loss, 
            ckpt_dir,
            tb_writer=None,
            **kwargs
    ):
        self.dro_loss = kldro_loss(kwargs['beta'], loss_fn)
        self.erm = ERM(device, model, optimizer, dataset, self.dro_loss, ckpt_dir, tb_writer, **kwargs)

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], **kwargs):
        return self.erm.train(num_training_updates, logging_steps, eval_steps, metrics, **kwargs)
    
    def evaluate(self, dataloader, metrics=[]):
        return self.erm.evaluate(dataloader, metrics)



# @trainer_register.register
# class WDRO(object):
   
#     def cost_function(self, x1, x2):
#         return ((x1-x2)*(x1-x2)).sum()

#     def __init__(self, 
#             device, 
#             model, 
#             optimizer, 
#             dataset, 
#             loss_fn:Loss, 
#             ckpt_dir, 
#             tb_writer=None,
#             **kwargs
#     ):
#         self.model = model.to(device)
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#         self.device = device
#         self.dataset = dataset

#         reset_parameters(self.model, kwargs['model_init'])

#     def attack(self, budget, data, steps=15):
#         """
#         Launch an adversarial attack using the Lagrangian relaxation.

#         Arguments:
#             budget: gamma in the original paper. Note that this parameter is
#                 different from the budget parameter in other DRO classes.
#         """

#         images, labels = data
#         images_adv = images.clone().detach().to(self.device)
#         images_adv.requires_grad_(True)

#         for i in range(steps):
#             if images_adv.grad is not None:
#                 images_adv.grad.data.zero_()
#             outputs = self.model(images_adv).squeeze()
#             loss = self.loss_fn(outputs, labels, use_weight=False, reduction='sum') \
#                  - budget * self.cost_function(images, images_adv)
#             loss.backward()
#             images_adv.data.add_(1 / np.sqrt(i + 1) * images_adv.grad)
#             images_adv.data.clamp_(0, 1)
#         return images_adv, labels

#     def train(self, 
#             num_training_updates,
#             logging_steps, 
#             eval_steps=0, 
#             metrics=[],
#             **kwargs
#     ):

#         steps_adv = kwargs['steps_adv']
#         budget = kwargs['budget']

#         iterator = iter(cycle(self.dataset.training_loader))
#         for i in tqdm(range(num_training_updates), desc='Training'):
#             images, labels, ids = next(iterator)
#             images = images.to(self.device)
#             labels = labels.to(self.device)     
#             data = (images, labels)

#             images_adv, labels = self.attack(budget, data, steps=steps_adv)

#             self.model.train()
#             self.optimizer.zero_grad()
#             outputs = self.model(images_adv).squeeze()
#             loss = self.loss_fn(outputs, labels)
#             loss.backward()
#             self.optimizer.step()
#             if eval_steps > 0 and (i+1) % eval_steps == 0:
#                 val_loss, metric_dict = self.evaluate(self.dataset.validation_loader, metrics)
#                 print (f'train loss:{loss} val loss:{val_loss}')
#                 print (metric_dict)
#             if logging_steps > 0 and (i+1) % logging_steps == 0:
#                 print (f'train loss:{loss}')
        
#         return num_training_updates - 1
                      
#     def evaluate(self, dataloader, metrics=[], loss_reduction=True):
#         self.model.eval()
#         sample = 0
#         loss_list = []
#         loss = 0
#         # correct_number = 0
#         predict_list = []
#         label_list = []
#         with torch.no_grad():
#             for bundle_batch in tqdm(dataloader, desc='Evaluating'):
#                 input_batch, label_batch = bundle_batch[0], bundle_batch[1]
#                 batch_size = input_batch.shape[0]
#                 input_batch = input_batch.to(self.device)
#                 label_batch = label_batch.to(self.device)
#                 sample += batch_size
#                 predict = self.model(input_batch).squeeze() # [batch_size]
                
#                 # calculate loss
#                 if loss_reduction:
#                     loss += self.loss_fn(predict, label_batch) * batch_size
#                 else:
#                     loss_list.append(self.loss_fn(predict, label_batch, reduction='none'))
#                 predict_list.append(predict)
#                 label_list.append(label_batch)
            
#             if loss_reduction:
#                 loss /= sample
#             else:
#                 loss = torch.cat(loss_list)
#             predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
#             label_array = torch.cat(label_list, dim=0).cpu().numpy()
        
#         if len(metrics)>0:
#             metric_dict = compute_metrics(predict_matrix, label_array, metrics=metrics)
#             return loss, metric_dict
#         else:  
#             return loss


@trainer_register.register
class CMixup(ERM):
    def __init__(self, device, model, optimizer, dataset:TensorLoader, loss_fn:Loss, regularizer:Loss, ckpt_dir, tb_writer=None, reset_model=True, **kwargs):
        super(CMixup, self).__init__(device, model, optimizer, dataset, loss_fn, regularizer, ckpt_dir, tb_writer, reset_model, **kwargs)
        self.sigma = kwargs['mixup_sigma']
        self.alpha = kwargs['mixup_alpha']
    
    def calculate_matrix(self, y):
        # Ensure y is a PyTorch tensor
        y = y.to(self._device)
        # Calculate the pairwise differences
        diff = y.unsqueeze(1) - y.unsqueeze(0)
        # Calculate the exponent term
        exponent = - (diff * diff) / (2 * self.sigma * self.sigma)
        # Calculate the matrix
        A = torch.exp(exponent)
        # row normalization
        # A = A / A.sum(dim=1, keepdim=True)
        return A 

    def mix(self, x, y, p):
        # Sample the index
        # print (p)
        idx = torch.cat([torch.multinomial(p[i], 1) for i in range(len(p))]) 
        # Sample the data
        x_sample = x[idx]
        y_sample = y[idx]
        # Sample from Beta(alpha, alpha), sample len(p) times to form 1-d tensor
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample((len(p), )).to(self._device)
        # Mixup
        x_mix = lam.reshape(-1, 1) * x_sample + (1 - lam).reshape(-1, 1) * x
        y_mix = lam * y_sample + (1 - lam) * y
        return x_mix, y_mix
    

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=[], start_step=0, **kwargs):
        _, y, _, _ = self._dataset.training_dataset.tensors
        P = self.calculate_matrix(y)

        iterator = iter(cycle(self._dataset.training_loader))
        min_valid_loss = float("inf")
        best_step = start_step + num_training_updates - 1
        for i in tqdm(range(num_training_updates), desc='Training'):
            self._model.train()
            # print (next(iterator))
            input_batch, label_batch, env_batch, ids = next(iterator)
            input_batch = input_batch.to(self._device)
            label_batch = label_batch.to(self._device).squeeze()
            env_batch = env_batch.to(self._device)
            ids = ids.to(self._device)

            P_batch = P[ids, :][:, ids]
            input_batch, label_batch = self.mix(input_batch, label_batch, P_batch)

            predict, phi, _ = self._model(input_batch)
            predict = predict.squeeze()
            loss = self._loss_fn(predict, label_batch, env_batch, use_weight=False, reduction='mean')
            loss.backward()
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
            
        return best_step
