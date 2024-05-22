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





parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1314, type=int) 
parser.add_argument('--cuda', default='cuda:0', type=str)
parser.add_argument('--do_train', action='store_false')
parser.add_argument('--checkpoint_dir', default=None, type=str)
parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--learning_rate', default=0.05, type=float) 
parser.add_argument('--drop_rate', default=0, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--bias', action='store_false')
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--hidden_dim', default=[32,8], type=int, nargs='+') 
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--dataset_path', default='./data/power/processed_test', type=str)
parser.add_argument('--split', default=['test'], type=str, nargs='+')
parser.add_argument('--iteration', default=200, type=int)
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
parser.add_argument('--mc_hclass', default='HardSampleHClass', type=str) # LogitDensityRatioHClass
parser.add_argument('--verbose', action='store_false')
parser.add_argument('--num_mc_updates', default=10, type=int)
parser.add_argument('--mc_round_interval', default=0, type=int) 
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--mc_finetune_steps', default=20, type=int)
parser.add_argument('--method_name', default='MCDeboost', type=str)
parser.add_argument('--domain_clf_lr', default=0.01, type=int)
parser.add_argument('--alpha_threshold', default=0.5, type=float)
parser.add_argument('--jtt_lambda_up', default=5, type=float)



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


device = torch.device(args.cuda if torch.cuda.is_available() and args.cuda != None else 'cpu')
print(f'device:{device}')

print ('Loading Dataset')
dataset = TensorLoader(batch_size=args.batch_size, path=args.dataset_path, split=args.split, workers=1)


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


if args.trainer == 'MCDeboost':
    trainer = trainer_register[args.trainer](device, model, optimizer, dataset, mse_loss_vanilla(), ckpt_dir, **args.__dict__)
elif args.trainer == 'GroupDRO' or args.trainer == 'X2DRO' or args.trainer == 'TERM':
    trainer = trainer_register[args.trainer](device, model, optimizer, dataset, mse_loss(), ckpt_dir, **args.__dict__)
else:
    trainer = trainer_register[args.trainer](device, model, optimizer, dataset, mse_loss(), reg_object, ckpt_dir, tb_writer, **args.__dict__)
# In[ ]:




best_step = None
if args.do_train:
    if ckpt_dir != None:
        json.dump(args.__dict__, open(f'{ckpt_dir}/argument.json','w'))
    if args.trainer == 'MCDeboost':
        train_metric_dict = trainer.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)
    else:
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
                label_batch = label_batch.to(device)
                predict, _, _ = model(input_batch) # [batch_size]
                predict = predict.squeeze()
                predict_list.append(predict)
                label_list.append(label_batch)
                env_list.append(env_batch)
                    
            predict_matrix = torch.cat(predict_list, dim=0).cpu().numpy()
            label_array = torch.cat(label_list, dim=0).cpu().numpy()        
            env_array = torch.cat(env_list, dim=0).cpu().numpy()    
        
        assert (len(metrics)>0)
        # metric_dict_list = []
        # for env in np.unique(env_array):
        #     metric_dict = compute_metrics(predict_matrix[env_array==env], label_array[env_array==env], metrics=metrics)
        #     metric_dict_list.append(metric_dict)
        
        average_metric_dict = compute_metrics(predict_matrix, label_array, metrics=metrics)

        return average_metric_dict

val_metric_dict = evaluate_env(dataset.validation_loader, model, device, metrics=args.metric_list)

      
import pandas as pd

if os.path.exists(f'power/results.csv'):
    df = pd.read_csv(f'power/results.csv', header=0)
else:
    df = pd.DataFrame(columns=['timestamp', 'method','RMSE_test','RMSE_val','hyperparameter', 'val_metric'])

hyperparameter = f'lr{args.learning_rate}_bs{args.batch_size}_reg{args.reg_lambda}_jtt{args.jtt_lambda_up}_alpha{args.alpha_threshold}'
val_metric = train_metric_dict['err_inv'] if args.trainer == 'MCDeboost' else val_metric_dict['RMSE']

df = df.append({
    'timestamp':pd.Timestamp.now(),
    'method':args.method_name,
    'RMSE_test':metric_dict['RMSE'],
    'RMSE_val':val_metric_dict['RMSE'],
    'seed':args.seed,
    'hyperparameter':hyperparameter,
    'val_metric':val_metric
}, ignore_index=True)
df.to_csv(f'power/results.csv', index=False)

# In[ ]:# In[ ]:



