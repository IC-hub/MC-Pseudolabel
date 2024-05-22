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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1314, type=int)
parser.add_argument('--no_cuda', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--checkpoint_dir', default=None, type=str)
parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--drop_rate', default=0, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--batchnorm', action='store_true')
parser.add_argument('--hidden_dim', default=[], type=int, nargs='+')
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--dataset_path', default='data/Ionosphere', type=str)
parser.add_argument('--split', default=[], type=str, nargs='+')
parser.add_argument('--iteration', default=200, type=int)
parser.add_argument('--log_steps', default=1, type=int)
parser.add_argument('--evaluation_steps', default=1, type=int)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--metric_list', default=[], type=str, nargs='+')
parser.add_argument('--k_neighbor', default=5, type=int)
parser.add_argument('--knn_mode', default='connectivity', type=str)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--flow_lr', default=10, type=float)
parser.add_argument('--flow_steps', default=10000, type=int)
parser.add_argument('--flow_eval_steps', default=1000, type=int)
parser.add_argument('--eta', default=0, type=float)
parser.add_argument('--model_init', default='default', type=str)
parser.add_argument('--trainer', default='ERM', type=str)
parser.add_argument('--loss', default='bce_loss', type=str)
parser.add_argument('--output_dim', default=1, type=int)
parser.add_argument('--nndescent', action='store_true')
parser.add_argument('--steps_adv', default=50, type=int)
parser.add_argument('--budget', default=1, type=float)




args = parser.parse_args()



def main(args):
    
    print (args)
    
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


    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'device:{device}')
    
    print ('Loading Dataset')
    dataset = TensorLoader(batch_size=args.batch_size, path=args.dataset_path, split=args.split, workers=1)

    model = MLP(input_dim = dataset.feature_dim, 
            hidden_size = args.hidden_dim, 
            drop_rate = args.drop_rate,
            batchnorm = args.batchnorm,
            output_features = args.output_dim
    )

    tb_writer = SummaryWriter(log_dir) if log_dir != None else None

    parameter_list_decay = []
    parameter_list_wo_decay = []
    for name, para in model.named_parameters():
        if 'bias' in name:
            parameter_list_wo_decay.append(para)
        else:
            parameter_list_decay.append(para)

    optimizer = optim.Adam([
                {'params': parameter_list_wo_decay},
                {'params':parameter_list_decay, 'weight_decay': args.weight_decay}
            ], 
            lr=args.learning_rate
    )

    loss_fn = loss_register[args.loss]()

    trainer = trainer_register[args.trainer](device, model, optimizer, dataset, loss_fn, ckpt_dir, tb_writer, **args.__dict__)

    best_step = None
    if args.do_train:
        if ckpt_dir != None:
            json.dump(args.__dict__, open(f'{ckpt_dir}/argument.json','w'))
        best_step = trainer.train(args.iteration, args.log_steps, args.evaluation_steps, args.metric_list, **args.__dict__)


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
        loss, metric_dict = trainer.evaluate(dataset.test_loader[group], args.metric_list)
        print(f'split:{group}\nloss:{loss}')
        print(metric_dict)
    



if __name__ == '__main__':
    main(args)
