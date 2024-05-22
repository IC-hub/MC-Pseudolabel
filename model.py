from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
from register import Register



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, drop_rate=0, batchnorm=False, output_features=1, bias=True):
        '''
        Args:
            hidden_size: list of hidden unit dimensions, the number of elements equal the humber of hidden layers.
        '''
        super(MLP, self).__init__()
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
            # Ativation except last layer
            if i < len(self.hidden_size)-2:
                if batchnorm:
                    self.hidden_layers.append((f'batchnorm{i+1}', nn.BatchNorm1d(num_features=output_dim)))
                self.hidden_layers.append((f'relu{i+1}', nn.ReLU()))
                self.hidden_layers.append((f'dropout{i+1}', nn.Dropout(p=drop_rate)))
        # output layer
        self.output_layer = nn.Linear(
                            in_features = self.hidden_size[-1],
                            out_features = output_features,
                            bias=bias
                        )
        self.hidden_layers.append((f'output', self.output_layer))
        print (self.hidden_layers)
        
        self.fc = nn.Sequential(OrderedDict(self.hidden_layers[:-1]))


    def forward(self, x):
        phi = self.fc(x)
        y = self.output_layer(phi)
        return y, phi
    
    def model(self):
        return self.fc

    def head(self):
        return self.output_layer

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

class MLP2Sigmoid(MLP2):
    def __init__(self, input_dim, hidden_size, drop_rate=0, batchnorm=False, output_features=1, bias=True):
        super(MLP2Sigmoid, self).__init__(input_dim, hidden_size, drop_rate, batchnorm, output_features, bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        phi = self.fc(x)
        y = self.output_layer(phi)
        return self.sigmoid(y), None, None
    
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
        # print (self.weight)

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
        # if self.weight is None or use_weight==False:
        #     return nn.BCEWithLogitsLoss(weight=None, reduction=reduction)(predict, target.float())
        # else:
        #     return nn.BCEWithLogitsLoss(weight=self.weight.to(predict), reduction=reduction)(predict, target.float())

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

@loss_register.register
class groupDRO(Loss):
    def __init__(self, risk:Loss, device, n_env=2, eta=0.01):
        super(groupDRO, self).__init__()
        self.risk = risk
        self.device = device
        self.n_env = n_env
        self.eta = eta
        self.prob = np.ones(n_env) / n_env
    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        loss0 = self.risk(predict, target, env, reduction='none')
        if reduction == 'none':
            return loss0
        loss_env = torch.zeros(self.n_env).to(self.device)
        for env_id in env.unique():
            loss_env[env_id] = loss0[env==env_id].mean()
            self.prob[env_id] = np.exp(self.eta*loss_env[env_id].item()) * self.prob[env_id]
        self.prob = self.prob / self.prob.sum()
        # print (self.prob)
        loss = 0
        for env_id in env.unique():
            loss += self.prob[env_id] * loss_env[env_id]
        return loss
        

        


def flatten_and_concat_variables(vs):
    """Flatten and concat variables to make a single flat vector variable."""
    flatten_vs = [torch.flatten(v) for v in vs]
    # print (flatten_vs)
    # if len(flatten_vs) > 1:
    return torch.cat(flatten_vs, axis=0)
    # else:
    #     return flatten_vs[0]

@loss_register.register
class MIP(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        loss = risk(predict, target, env, reduction='none')
        loss_env = []
        for env_id in env.unique():
            loss_env.append(loss[env==env_id].mean())
        grad_env = []
        for per_loss in loss_env:
            per_grad = torch.autograd.grad(per_loss, network.parameters(), retain_graph=True, create_graph=True)
            grad_env.append(flatten_and_concat_variables(per_grad))
        grad_env = torch.stack(grad_env, dim=0)
        # print (grad_env)
        # print (torch.var(grad_env, dim=0))
        if len(env.unique()) == 1:
            grad_var = 0
        else:
            grad_var = torch.var(grad_env, dim=0).mean()
        # grad_mean = torch.mean(torch.abs(grad_env))
        return grad_var 

@loss_register.register
class CLOvE(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        loss_env = []
        for env_id in env.unique():
            predict_env = predict[env==env_id]
            target_env = target[env==env_id]
            loss_env.append(self.calibration_unbiased_loss(predict_env, target_env))
        loss_env = torch.stack(loss_env, dim=0)
        loss = loss_env.mean()
        return loss 
    
    def calibration_unbiased_loss(self, logits, correct_labels):
        """Function to compute MMCE_m loss in PyTorch."""
        c_minus_r = correct_labels.float() - logits
        dot_product = torch.matmul(c_minus_r.unsqueeze(1), c_minus_r.unsqueeze(1).transpose(0, 1))
        
        prob_tiled = logits.unsqueeze(1).repeat(1, logits.shape[0]).unsqueeze(2)
        prob_pairs = torch.cat((prob_tiled, prob_tiled.transpose(0, 1)), dim=2)
        
        def torch_kernel(matrix):
            return torch.exp(-1.0 * torch.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (2 * 0.2))

        kernel_prob_pairs = torch_kernel(prob_pairs)
        numerator = dot_product * kernel_prob_pairs
        return torch.sum(numerator) / (logits.shape[0]**2)

@loss_register.register
class IDGM(Loss):
    ''' Gradient Matching for Domain Generalisation '''
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        loss = risk(predict, target, env, reduction='none')
        loss_env = []
        for env_id in env.unique():
            loss_env.append(loss[env==env_id].mean())
        grad_env = []
        for per_loss in loss_env:
            per_grad = torch.autograd.grad(per_loss, network.parameters(), retain_graph=True, create_graph=True)
            grad_env.append(flatten_and_concat_variables(per_grad))
        # grad_env = torch.stack(grad_env, dim=0)
        if len(env.unique()) == 1:
            inner_prod = 0
        elif len(env.unique()) == 2:
            inner_prod = - (grad_env[0] * grad_env[1]).mean()
        elif len(env.unique() == 3):
            inner_prod = - 1/3 * (grad_env[0]*grad_env[1] + grad_env[0]*grad_env[2] + grad_env[1]*grad_env[2]).mean()
        else:
            raise NotImplementedError
        return inner_prod

@loss_register.register
class IRM(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        loss = risk(predict, target, env, reduction='none')
        loss_env = []
        for env_id in env.unique():
            loss_env.append(loss[env==env_id].mean())
        grad_env = []
        for per_loss in loss_env:
            per_grad = torch.autograd.grad(per_loss, network.parameters(), retain_graph=True, create_graph=True)
            grad_env.append(flatten_and_concat_variables(per_grad))
        grad_env = torch.stack(grad_env, dim=0)
        # print (grad_env)
        # print (torch.var(grad_env, dim=0))
        grad_var = (grad_env * grad_env).mean()
        # grad_mean = torch.mean(torch.abs(grad_env))
        return grad_var 

@loss_register.register
class IBIRM(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        irm = IRM()(predict, phi, target, env, network, risk)
        var = torch.var(phi, dim=0).mean()
        loss = irm + var
        return loss


@loss_register.register
class MIPNORM(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        loss = risk(predict, target, env, reduction='none')
        loss_env = []
        for env_id in env.unique():
            loss_env.append(loss[env==env_id].mean())
        grad_env = []
        for per_loss in loss_env:
            per_grad = torch.autograd.grad(per_loss, network.parameters(), retain_graph=True, create_graph=True)
            grad_env.append(flatten_and_concat_variables(per_grad))
        grad_env = torch.stack(grad_env, dim=0)
        # print (grad_env)
        # print (torch.var(grad_env, dim=0))
        grad_var = torch.var(grad_env, dim=0).mean()
        # grad_mean = torch.mean(torch.abs(grad_env))
        loss_mean = torch.mean(torch.stack(loss_env, dim=0))
        reg_env = []
        for i, per_loss in enumerate(loss_env):
            reg_env.append(grad_env[i].mean())
        reg_env = torch.stack(reg_env, dim=0)
        return grad_var / (reg_env * reg_env).mean()

@loss_register.register
class REX(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        loss = risk(predict, target, env, reduction='none')
        loss_env = []
        for env_id in env.unique():
            loss_env.append(loss[env==env_id].mean())
        loss_env = torch.stack(loss_env, dim=0)
        # print (grad_env)
        # print (torch.var(grad_env, dim=0))
        if len(env.unique()) == 1:
            var = 0
        else:
            var = torch.var(loss_env)
        # mean = torch.mean(loss_env)
        return var

@loss_register.register
class REXNORM(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        loss = risk(predict, target, env, reduction='none')
        loss_env = []
        for env_id in env.unique():
            loss_env.append(loss[env==env_id].mean())
        loss_env = torch.stack(loss_env, dim=0)
        # print (grad_env)
        # print (torch.var(grad_env, dim=0))
        var = torch.var(loss_env)
        mean = torch.mean(loss_env)
        return var / (mean*mean)
        

@loss_register.register
class SIM(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss, alpha=1):
        loss = MIP()(predict, phi, target, env, network, risk)
        grad = torch.autograd.grad(loss, network.parameters(), retain_graph=True, create_graph=True)
        grad = flatten_and_concat_variables(grad)
        # grad = (grad*grad).mean()
        # return grad

        loss = risk(predict, target, env, reduction='none')
        loss_env = []
        value_env = []
        for env_id in env.unique():
            loss_env.append(loss[env==env_id].mean())
            value_env.append(phi[env==env_id].mean(dim=0))
        A_env = []
        for k, per_loss in enumerate(loss_env):
            per_grad = torch.autograd.grad(per_loss, network.parameters(), retain_graph=True, create_graph=True)
            per_grad = flatten_and_concat_variables(per_grad)
            hessian = torch.zeros(per_grad.shape[0], per_grad.shape[0]).to(predict)
            for i in range(per_grad.shape[0]):
                grad2 = torch.autograd.grad(per_grad[i], network.parameters(), retain_graph=True, create_graph=True)
                grad2 = flatten_and_concat_variables(grad2)
                hessian[i] = grad2
            A = torch.eye(per_grad.shape[0]).to(predict) - alpha * hessian
            A = torch.matmul(A, value_env[k].reshape(-1, 1)).squeeze()
            A_env.append(A)
        A = torch.stack(A_env, dim=1)

        ATA_inv = torch.linalg.inv(A.t() @ A)
        grad = grad - (A @ ATA_inv @ A.t() @ grad.reshape(-1, 1)).squeeze()
        grad = (grad*grad).mean()
        return grad

    def project_onto_left_null_space(A, x):
        # Ensure gradient computations are off (usually faster and saves memory)
        with torch.no_grad():
            # Compute the SVD of A transpose
            U, _, _ = torch.svd(A.t())
            
            # The number of vectors in the left null space
            # is the difference between the total columns of A
            # and its rank.
            rank = min(A.size())
            
            # Extract the vectors that span the left null space
            basis_left_null_space = U[:, rank:]
            
            # Project x onto each of these vectors and sum up the projections
            projection = torch.zeros_like(x)
            for i in range(basis_left_null_space.size(1)):
                w = basis_left_null_space[:, i]
                projection += torch.dot(x, w) / torch.dot(w, w) * w
                
        return projection

@loss_register.register
class SIM2(Loss):
    def __call__(self, predict, phi, predict_baseline, target, target_baseline, env, head:nn.Module, baseline:nn.Module, risk:Loss):
        loss = MIP()(predict, phi, target, env, head, risk)
        loss_baseline = MIP()(predict_baseline, phi, target_baseline, env, baseline, risk)
        # loss_baseline = 0
        return loss, loss_baseline

@loss_register.register
class SIM2NORM(Loss):
    def __call__(self, predict, phi, predict_baseline, target, target_baseline, env, head:nn.Module, baseline:nn.Module, risk:Loss):
        loss = MIPNORM()(predict, phi, target, env, head, risk)
        loss_baseline = MIPNORM()(predict_baseline, phi, target_baseline, env, baseline, risk)
        # loss_baseline = 0
        return loss, loss_baseline

@loss_register.register
class SIMREX(Loss):
    def __call__(self, predict, phi, predict_baseline, target, target_baseline, env, head:nn.Module, baseline:nn.Module, risk:Loss):
        loss = REXNORM()(predict, phi, target, env, head, risk)
        loss_baseline = REXNORM()(predict_baseline, phi, target_baseline, env, baseline, risk)
        # loss_baseline = 0
        return loss, loss_baseline

@loss_register.register
class MRI(Loss):
    def __call__(self, predict, phi, target, env, network:nn.Module, risk:Loss):
        if risk.__class__.__name__ == 'mse_loss':
            # target_tensor = nn.Parameter(target)
            # loss = risk(predict, target_tensor, env, reduction='none')
            grad_env = []
            for env_id in env.unique():
                per_target, per_predict = target[env==env_id], predict[env==env_id]
                per_grad = 2 * (per_target - per_predict)
                per_grad = per_grad * per_target
                grad_env.append(per_grad.mean())
            if len(env.unique()) == 1:
                return 0
            if len(env.unique()) == 2:
                return 0.5 * (grad_env[0] - grad_env[1]) * (grad_env[0] - grad_env[1])
            elif len(env.unique()) == 3:
                return 0.5 * (grad_env[0] - grad_env[1]) * (grad_env[0] - grad_env[1]) \
                    + 1/6 * (grad_env[0] + grad_env [1] - 2*grad_env[2]) * (grad_env[0] + grad_env [1] - 2*grad_env[2])
            else:
                print (f'Invalid env number: {len(env.unique())}')
                raise NotImplementedError
        elif risk.__class__.__name__ == 'bce_loss':
            assert (len(env.unique()) == 2)
            grad_env = []
            for env_id in env.unique():
                per_target, per_predict = target[env==env_id], predict[env==env_id]
                per_grad = - per_predict
                per_grad = per_grad * per_target
                grad_env.append(per_grad.mean())
            return 0.5 * (grad_env[0] - grad_env[1]) * (grad_env[0] - grad_env[1])
        else:
            raise NotImplementedError

@loss_register.register
class x2dro_loss(Loss):
    def __init__(self, risk:Loss, eta):
        super(x2dro_loss, self).__init__(weight=None)
        self.eta = eta
        self.loss_fn = risk

    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        '''
        Args:
            reduction and useweight is useless.
        '''
        loss = self.loss_fn(predict, target, env, reduction='none')
        loss = loss - self.eta
        loss = torch.nn.functional.relu(loss)
        loss = torch.sqrt(torch.mean(loss*loss))
        return loss       
    
@loss_register.register
class tilted_loss(Loss):
    def __init__(self, risk:Loss, t):
        super(tilted_loss, self).__init__(weight=None)
        self.t = t
        self.loss_fn = risk

    def __call__(self, predict, target, env, reduction='mean', use_weight=False):
        '''
        Args:
            reduction and useweight is useless.
        '''
        loss = self.loss_fn(predict, target, env, reduction='none')
        # return loss.mean()
        # print (loss, self.t)
        w = torch.exp(loss * self.t)
        tloss = (loss*w).mean() / w.mean()
        
        if torch.isinf(tloss) or torch.isnan(tloss):
            return loss.mean()

        # loss = self.loss_fn(predict, target, env, reduction='none')
        # # return loss.mean()
        # # print (loss, self.t)
        # tloss = torch.exp(loss * self.t)
        # # print (loss)
        # inf_mask = ~torch.isinf(tloss)
        # if len(inf_mask) == 0:
        #     print (loss.mean())
        #     return loss.mean()
        # else:
        #     tloss = tloss[inf_mask].mean()
        #     tloss = torch.log(tloss) / self.t
        #     if torch.isinf(tloss) or torch.isnan(tloss):
        #         print (loss.mean())
        #         return loss.mean()
        # print (tloss)
        
        

        return tloss



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
    # return os.path.split(path)[0]
    return path

