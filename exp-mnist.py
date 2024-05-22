#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



# In[3]:


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1314, type=int) 
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--do_train', action='store_false')
parser.add_argument('--checkpoint_dir', default=None, type=str)
parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--learning_rate', default=0.005, type=float) 
parser.add_argument('--learning_rate_head', default=0.005, type=float)
parser.add_argument('--drop_rate', default=0, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--hidden_dim', default=[64,16], type=int, nargs='+')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--dataset_path', default='data/ColoredMNIST/data/ColoredMNIST', type=str)
parser.add_argument('--split', default=['test'], type=str, nargs='+')
parser.add_argument('--iteration', default=400, type=int)
parser.add_argument('--log_steps', default=20, type=int)
parser.add_argument('--evaluation_steps', default=20, type=int)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--metric_list', default=['Accuracy','AUC','F1','F1_macro'], type=str, nargs='+')
parser.add_argument('--k_neighbor', default=5, type=int)
parser.add_argument('--knn_mode', default='connectivity', type=str)
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--flow_lr', default=10, type=float)
parser.add_argument('--flow_steps', default=300, type=int)
parser.add_argument('--flow_eval_steps', default=200, type=int)
parser.add_argument('--eta', default=0, type=float)
parser.add_argument('--model_init', default='default', type=str)
parser.add_argument('--reload_best', action='store_true')
parser.add_argument('--trainer', default='ERM', type=str)
parser.add_argument('--reg_name', default=None, type=str)
parser.add_argument('--reg_lambda', default=200, type=float)
parser.add_argument('--tune_head', action='store_true')





args = parser.parse_args()

    
   


# In[4]:


print (args)


# In[5]:


# For reproduction
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

def run_exp():

    ckpt_dir = os.path.join(args.checkpoint_dir, args.name) if args.checkpoint_dir != None else None
    if ckpt_dir != None:
        os.makedirs(ckpt_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, args.name) if args.log_dir != None else None


    device = torch.device('cuda:1' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'device:{device}')

    print ('Loading Dataset')
    dataset = TensorLoader(batch_size=args.batch_size, path=args.dataset_path, split=args.split, workers=1)


    # In[6]:


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
        parameter_list_wo_decay, parameter_list_decay = get_parameters([model.model(), model.head()])
    else:
        parameter_list_wo_decay, parameter_list_decay = get_parameters([model.model()])
    optimizer = optim.Adam([
                {'params': parameter_list_wo_decay},
                {'params':parameter_list_decay, 'weight_decay': args.weight_decay}
            ], 
            lr=args.learning_rate
    )

    parameter_list_wo_decay, parameter_list_decay = get_parameters([model.baseline()])
    optimizer_baseline = optim.Adam([
                {'params': parameter_list_wo_decay},
                {'params':parameter_list_decay, 'weight_decay': args.weight_decay}
            ], 
            lr=args.learning_rate_head
    )

    if args.reg_name is None:
        reg_object = None
    else:
        reg_object = loss_register[args.reg_name]()
    
    
    
    # In[7]
    if args.trainer == 'ERM':
        trainer = trainer_register[args.trainer](device, model, optimizer, dataset, bce_loss(), reg_object, ckpt_dir, tb_writer, **args.__dict__)

    elif args.trainer == 'ERM_SIM':
        # trainer_ERM = trainer_register['ERM'](device, model, optimizer, dataset, bce_loss(), None, ckpt_dir, tb_writer, **args.__dict__)
        trainer = trainer_register[args.trainer](device, model, optimizer, optimizer_baseline, dataset, bce_loss(), reg_object, ckpt_dir, tb_writer, **args.__dict__)
    
    elif args.trainer == 'groupDRO':
        trainer = trainer_register['ERM'](device, model, optimizer, dataset, groupDRO(bce_loss(), device, n_env=2, eta=0.01), reg_object, ckpt_dir, tb_writer, **args.__dict__)    
    
    else:
        raise NotImplementedError
    # In[29]:



    best_step = None
    if args.do_train:
        if ckpt_dir != None:
            json.dump(args.__dict__, open(f'{ckpt_dir}/argument.json','w'))
        best_step = trainer.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)
        # if args.trainer == 'ERM':
        #     best_step = trainer.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)
        # elif args.trainer == 'ERM_SIM':
        #     # best_step = trainer_ERM.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)
        #     # trainer = trainer_register[args.trainer](device, model, optimizer, optimizer_baseline, dataset, bce_loss(), reg_object, ckpt_dir, tb_writer,reset_model=False, **args.__dict__)
        #     best_step = trainer.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)


    # In[21]:

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

    metric_dict_final = {}
    for group in args.split:
        print(f'split:{group}')
        metric_dict = trainer.evaluate(dataset.test_loader[group], args.metric_list, return_loss=False)
        # metric_dict = trainer.evaluate(dataset.validation_loader, args.metric_list, return_loss=False)
        for k,v in metric_dict.items():
            metric_dict_final[k+'_'+group]  = v
    
    print(metric_dict_final)
    return metric_dict_final
  


metric_dict_list = {}
for _ in range(3):
    metric_dict = run_exp()
    for k,v in metric_dict.items():
        if k in metric_dict_list:
            metric_dict_list[k].append(v)
        else:
            metric_dict_list[k] = [v]

metric_result = {}
for k,v in metric_dict_list.items():
    metric_result[k+'mean'] = np.mean(v)
    metric_result[k+'std'] = np.std(v)
print (metric_result)


