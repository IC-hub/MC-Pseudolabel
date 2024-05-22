#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import argparse
import random
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter

from dataset import *
from trainer import *
from model import *
from hclass import *


# In[7]:


def generate_new_toy_example():
    S = np.random.normal(0, 1, [10000, 1])
    V = np.random.normal(0, 1, [10000, 1])
    # Y = np.zeros((100, 1))
    noise = np.random.normal(0, 0.5, [10000, 1])
    Y = S
    Y = Y + noise
    noise = np.random.normal(0, 0.1, [10000, 1])
    V[:5000] = 1.25*Y[:5000]
    V[5000:] = 0.75*Y[5000:]
    V = V + noise
    Y = Y.reshape(-1)
    E = np.zeros(10000)
    E[5000:] = 1


    # noise_list = np.random.randint(0, 9500, 20)
    # Y[noise_list] += 20
    return torch.tensor(np.hstack((S, V)), dtype=torch.float),             torch.tensor(Y, dtype=torch.float),             torch.tensor(E, dtype=torch.long)

def generate_toy_data(alpha):
    S = np.random.normal(0, 1, [2000, 1])
    V = np.random.normal(0, 1, [2000, 1])
    noise = np.random.normal(0, 0.5, [2000, 1])
    Y = S
    Y = Y + noise
    noise = np.random.normal(0, 0.1, [2000, 1])
    V = Y*alpha
    V = V + noise
    Y = Y.reshape(-1)
    return torch.tensor(np.hstack((S, V)), dtype=torch.float),             torch.tensor(Y, dtype=torch.float)


# In[8]:


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1314, type=int) 
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--do_train', action='store_false')
parser.add_argument('--checkpoint_dir', default=None, type=str)
parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--learning_rate', default=0.1, type=float) 
parser.add_argument('--drop_rate', default=0, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--bias', action='store_false')
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--hidden_dim', default=[], type=int, nargs='+')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--dataset_path', default=None, type=str)
parser.add_argument('--split', default=['-1'], type=str, nargs='+')
parser.add_argument('--iteration', default=100, type=int)
parser.add_argument('--log_steps', default=20, type=int)
parser.add_argument('--evaluation_steps', default=20, type=int)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--metric_list', default=['RMSE'], type=str, nargs='+')
parser.add_argument('--model_init', default='default', type=str)
parser.add_argument('--reload_best', action='store_true')
parser.add_argument('--trainer', default='MCDeboost', type=str) # MCDeboost
parser.add_argument('--reg_lambda', default=1, type=float)
parser.add_argument('--reg_name', default=None, type=str)
parser.add_argument('--tune_head', action='store_true')
parser.add_argument('--mc_hclass', default='NeuralDensityRatioHClass', type=str) # LogitDensityRatioHClass
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--num_mc_updates', default=20, type=int)
parser.add_argument('--mc_round_interval', default=0, type=int) 
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--mc_finetune_steps', default=20, type=int)
parser.add_argument('--method_name', default='MCDeboost', type=str)



args = parser.parse_args()



    
   


# In[9]:


print (args)


# In[10]:


# For reproduction
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

ckpt_dir = os.path.join(args.checkpoint_dir, args.name) if args.checkpoint_dir != None else None
if ckpt_dir != None:
    os.makedirs(ckpt_dir, exist_ok=True)
log_dir = os.path.join(args.log_dir, args.name) if args.log_dir != None else None


device = torch.device('cuda:1' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
print(f'device:{device}')

print ('Loading Dataset')
dataset = TensorLoader(batch_size=args.batch_size, path=args.dataset_path, split=args.split, workers=1,
                      data_tensors=(
                          generate_new_toy_example(),
                          generate_new_toy_example(),
                          generate_toy_data(-1)
                      ))
# In[ ]:





# In[11]:


model = MLP2(input_dim = dataset.feature_dim, 
        hidden_size = args.hidden_dim, 
        drop_rate = args.drop_rate,
        batchnorm = args.batchnorm,
        bias = args.bias
)


tb_writer = SummaryWriter(log_dir) if log_dir != None else None

def get_parameters(network_list):
    parameter_list_decay = []
    parameter_list_wo_decay = []
    for network in network_list:
        for name, para in network.named_parameters():
            if 'bias' in name:
                parameter_list_wo_decay.append(para)
            else:
                parameter_list_decay.append(para)
    return parameter_list_decay, parameter_list_wo_decay


if args.tune_head:
    parameter_list_wo_decay, parameter_list_decay = get_parameters([model.model(),model.head()])
else:
    parameter_list_wo_decay, parameter_list_decay = get_parameters([model.model()])


optimizer = optim.Adam([
            {'params': parameter_list_wo_decay},
            {'params':parameter_list_decay, 'weight_decay': args.weight_decay}
        ], 
        lr=args.learning_rate
)

if args.reg_name is None:
    reg_object = None
else:
    reg_object = loss_register[args.reg_name]()

if args.trainer == 'ERM':
    trainer = trainer_register[args.trainer](device, model, optimizer, dataset, mse_loss(), reg_object, ckpt_dir, tb_writer, **args.__dict__)
elif args.trainer == 'MCDeboost':
    trainer = trainer_register[args.trainer](device, model, optimizer, dataset, mse_loss_vanilla(), ckpt_dir, **args.__dict__)
elif args.trainer == 'GroupDRO':
     trainer = trainer_register[args.trainer](device, model, optimizer, dataset, mse_loss(), ckpt_dir, **args.__dict__)
else:
    raise NotImplementedError

# In[ ]:




best_step = None
if args.do_train:
    if ckpt_dir != None:
        json.dump(args.__dict__, open(f'{ckpt_dir}/argument.json','w'))
    best_step = trainer.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)


# In[ ]:


best_step = None # load recent epoch
# restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
if ckpt_dir != None:
    test_ckpt_dir = None
    if best_step is None:
        test_ckpt_dir = ckpt_dir
    else:
        test_ckpt_dir = ckpt_dir + f'/{best_step}' 
    dirname = restore(model, test_ckpt_dir) # + args.name)
    print (f'restore:{dirname}')

for group in args.split:
    metric_dict = trainer.evaluate(dataset.test_loader[group], args.metric_list, return_loss=False)
    print(f'split:{group}')
    print(metric_dict)

import pandas as pd

if os.path.exists(f'toy_slope_2D/results.csv'):
    df = pd.read_csv(f'toy_slope_2D/results.csv', header=0)
else:
    df = pd.DataFrame(columns=['timestamp', 'method','RMSE','hyperparameter'])

df = df.append({
    'timestamp':pd.Timestamp.now(),
    'method':args.method_name,
    'RMSE':metric_dict['RMSE'],
    'hyperparameter':args.reg_lambda
}, ignore_index=True)
df.to_csv(f'toy_slope_2D/results.csv', index=False)

# In[ ]:# In[ ]:



