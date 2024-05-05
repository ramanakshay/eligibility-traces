import torch
from torch.optim import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        # check boundaries
        defaults={'lr': lr}
        super().__init__(params, defaults)
    
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data -= group['lr']*grad
                
                
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.99,0.999), eps=1e-8):
        # check boundaries
        defaults={'lr': lr, 'beta1': betas[0], 'beta2': betas[1], 'eps':eps}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(exp_avg=torch.zeros_like(p.data),
                                    exp_avg_sq=torch.zeros_like(p.data),
                                    step=0)

    def step(self):
        for group in self.param_groups:
            for p in group['params']:

                state = self.state[p]
                grad = p.grad.data

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']
                state['step'] += 1

                exp_avg = torch.mul(exp_avg, beta1) + (1 - beta1)*grad
                exp_avg_sq = torch.mul(exp_avg_sq, beta2) + (1-beta2)*(grad*grad)

                momentum = exp_avg/(1-beta1**state['step'])
                variance = exp_avg_sq/(1-beta2**state['step'])
                std_dev = torch.sqrt( exp_avg_sq + group['eps'] )

                p.data -= group['lr'] * momentum / std_dev

                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq


class ETSGD(Optimizer):
    def __init__(self, params, lr=1e-3, gam=0.99, lam=0.9, clip=0.2, replacing=False):
            # check boundaries
            defaults={'lr': lr, 'gam': gam, 'lam': lam, 'clip': clip, 'replacing': replacing}
            super().__init__(params, defaults)
            for group in self.param_groups:
                for p in group['params']:
                    self.state[p] = dict(trace=torch.zeros_like(p.data),
                                        history=torch.zeros_like(p.data),
                                        step=0)
    
    def set_trace(self):
        # NOTE: This zeros out the grads as well
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                grad, trace = p.grad.data, state['trace']
                trace = group['lam'] * group['gam'] * trace + grad
                if group['replacing']:
                    trace = torch.clamp(trace, 1-group['clip'], 1+group['clip'])
                p.grad.data = torch.zeros_like(p.data)
                state['trace'] = trace
                
    def reset_trace(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trace'] = torch.zeros_like(p.data)
                
    def broadcast(self, residual):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                grad = p.grad.data
                trace, history = state['trace'], state['history']
                history += residual * trace
                state['history'] = history          
                        
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] += 1
                p.data += group['lr']*state['history']
                state['history'] = torch.zeros_like(p.data)
                
class ETAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.99,0.999), eps=1e-8, gam=0.99, lam=0.9, clip=0.2, replacing=False):
        # check boundaries
        defaults={'lr': lr, 'beta1': betas[0], 'beta2': betas[1], 'eps':eps,
                    'gam': gam, 'lam': lam, 'clip': clip, 'replacing': replacing}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(exp_avg=torch.zeros_like(p.data),
                                    exp_avg_sq=torch.zeros_like(p.data),
                                    trace=torch.zeros_like(p.data),
                                    history=torch.zeros_like(p.data),
                                    step=0)
    
    def set_trace(self):
        # NOTE: This zeros out the grads as well
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                grad, trace = p.grad.data, state['trace']
                trace = group['lam'] * group['gam'] * trace + grad
                if group['replacing']:
                    trace = torch.clamp(trace, -group['clip'], +group['clip'])
                p.grad.data = torch.zeros_like(p.data)
                state['trace'] = trace
                
    def reset_trace(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trace'] = torch.zeros_like(p.data)
                
    def broadcast(self, residual):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                grad = p.grad.data
                trace, history = state['trace'], state['history']
                history += residual * trace
                state['history'] = history
                
    def step(self):
        for group in self.param_groups:
            for p in group['params']:

                state = self.state[p]
                grad = state['history']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['beta1'], group['beta2']
                state['step'] += 1

                exp_avg = torch.mul(exp_avg, beta1) + (1 - beta1)*grad
                exp_avg_sq = torch.mul(exp_avg_sq, beta2) + (1-beta2)*(grad*grad)

                momentum = exp_avg/(1-beta1**state['step'])
                variance = exp_avg_sq/(1-beta2**state['step'])
                std_dev = torch.sqrt( exp_avg_sq + group['eps'] )

                p.data += group['lr'] * momentum / std_dev

                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq
                state['history'] = torch.zeros_like(p.data)
    