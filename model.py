from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
from register import Register


class Output(nn.Module):
    def __init__(self, input_dim, output_dim, init=1, bias=False):
        super(Output, self).__init__()
        assert (input_dim >= output_dim)
        # data = torch.ones(output_dim, input_dim)
        data = torch.zeros(output_dim, input_dim)
        for i in range(output_dim):
            data[i, i] = init
        self.weight = nn.Parameter(data)
        self.is_bias = bias
        if self.is_bias:
            data = torch.zeros(output_dim)
            self.bias = nn.Parameter(data)
        
    
    def forward(self, x):
        if self.is_bias:
            out = torch.mm(x, self.weight.t()) + self.bias
        else:
            out = torch.mm(x, self.weight.t())
        return out



class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_size, drop_rate=0, batchnorm=False, output_features=1, bias=True):
        '''
        Args:
            hidden_size: list of hidden unit dimensions, the number of elements equal the humber of hidden layers.
        '''
        super(MLP2, self).__init__()
        self.hidden_size = [input_dim] + hidden_size # self.hidden_size: [input dim, hidden dims, ...]
        self.hidden_layers = []
        # input layer and hidden layers
        for i in range(len(self.hidden_size) - 1):
            input_dim = self.hidden_size[i]
            output_dim = self.hidden_size[i + 1]
            self.hidden_layers.append((f'linear{i+1}', nn.Linear(
                            in_features = input_dim,
                            out_features = output_dim,
                            bias=bias
                        )))
            if batchnorm:
                self.hidden_layers.append((f'batchnorm{i+1}', nn.BatchNorm1d(num_features=output_dim)))
            self.hidden_layers.append((f'relu{i+1}', nn.ReLU()))
            self.hidden_layers.append((f'dropout{i+1}', nn.Dropout(p=drop_rate)))
        self.hidden_layers.append((f'linear{len(self.hidden_size)}', nn.Linear(
                            in_features = self.hidden_size[-1],
                            out_features = output_features,
                            bias=bias
                        )))
        # output layer
        self.output_layer = Output(output_features, output_features, init=1)
        self.baseline_layer = Output(output_features, output_features, init=0)
        print (self.hidden_layers)
        
        self.fc = nn.Sequential(OrderedDict(self.hidden_layers))


    def forward(self, x):
        phi = self.fc(x)
        y = self.output_layer(phi)
        y_baseline = self.baseline_layer(phi)
        return y, phi, y_baseline
    
    def model(self):
        return self.fc

    def head(self):
        return self.output_layer
     
    def baseline(self):
        return self.baseline_layer
    
def reset_parameters(module:nn.Module, method='default'):
    if method=='default':
        for layer in module.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                reset_parameters(layer, method)
    elif method=='normal':
        for p in module.parameters():
            nn.init.normal_(p)
    elif method=='constant':
        for p in module.parameters():
            nn.init.constant_(p, 1)
        
    else:
        raise NotImplementedError


global loss_register
loss_register = Register('loss_register')


class Loss:
    def __init__(self, weight=None):
        self.weight = weight
    
    def update_weight(self, weight):
        '''
        Args:
            weight: tensor or None
        '''
        self.weight = weight

    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        raise NotImplementedError

@loss_register.register
class bce_loss(Loss):
    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        loss = nn.BCEWithLogitsLoss(reduction='none')(predict, target.float())
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            total_loss = 0
            env_list = env.unique()
            for env_id in env_list:
                total_loss += loss[env==env_id].mean()
            total_loss /= len(env_list)
            return total_loss
        else:
            raise NotImplementedError

@loss_register.register
class bce_loss_vanilla(Loss):
    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        loss = nn.BCEWithLogitsLoss(reduction='none')(predict, target.float())
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise NotImplementedError

@loss_register.register
class ce_loss(Loss):
    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        if self.weight is None or use_weight==False:
            return nn.CrossEntropyLoss(weight=None, reduction=reduction)(predict, target.long())
        else:
            return nn.CrossEntropyLoss(weight=self.weight.to(predict), reduction=reduction)(predict, target.long())
        
@loss_register.register
class ce_loss_vanilla(Loss):
    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        loss = nn.CrossEntropyLoss(reduction='none')(predict, target.long())
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise NotImplementedError        

@loss_register.register
class mse_loss(Loss):
    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        loss = nn.MSELoss(reduction='none')(predict, target)
        # print (predict, target)
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            total_loss = 0
            env_list = env.unique()
            for env_id in env_list:
                total_loss += loss[env==env_id].mean()
            total_loss /= len(env_list)
            return total_loss
        else:
            raise NotImplementedError

@loss_register.register
class mse_loss_vanilla(Loss):
    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        loss = nn.MSELoss(reduction='none')(predict, target.float())
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise NotImplementedError


def save(model, ckpt_dir, global_step):
    os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
    path = os.path.join(ckpt_dir, str(global_step), 'pytorch_model.bin')
    torch.save(model.state_dict(), path)
    return os.path.split(path)[0]

def restore(model, ckpt_dir):
    try:
        if os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')):
            path = os.path.join(ckpt_dir, 'pytorch_model.bin')
        else:
            path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'pytorch_model.bin')
    except:
        print ('Model checkpoint unfound.')
        return None
    model.load_state_dict(torch.load(path))
    return path

