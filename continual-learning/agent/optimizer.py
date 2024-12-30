import torch
from torch.optim import Optimizer
import numpy as np

class SGDET(Optimizer):
    def __init__(self, params, config):
        # check boundaries
        defaults={'lr': config.lr, 'gam': config.gam, 'lam': config.lam}
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(trace=torch.zeros_like(p.data),
                                     history=torch.zeros_like(p.data),
                                     step=0)

    def set(self):
        # NOTE: This zeros out the grads as well
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                grad, trace = p.grad.data, state['trace']
                trace = group['lam'] * group['gam'] * trace + grad
                # if group['replacing']:
                #     trace = torch.clamp(trace, 1-group['clip'], 1+group['clip'])
                p.grad.data = torch.zeros_like(p.data)
                state['trace'] = trace

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trace'] = torch.zeros_like(p.data)

    def broadcast(self, residual):
        # residual is a scalar
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                trace, history = state['trace'], state['history']
                history += residual * trace
                state['history'] = history
        self.step()

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] += 1
                p.data += group['lr']*state['history']
                state['history'] = torch.zeros_like(p.data)


class AdamET(Optimizer):
    def __init__(self, params, config):
        super().__init__(params, config)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(exp_avg=torch.zeros_like(p.data),
                                     exp_avg_sq=torch.zeros_like(p.data),
                                     trace=torch.zeros_like(p.data),
                                     history=torch.zeros_like(p.data),
                                     step=0)

    def set(self):
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

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trace'] = torch.zeros_like(p.data)

    def broadcast(self, residual):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                trace, history = state['trace'], state['history']
                history += residual * trace
                state['history'] = history
        self.step()

    def step(self):
        for group in self.param_groups:
            for p in group['params']:

                state = self.state[p]
                grad = state['history']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas'][0], group['betas'][1]
                state['step'] += 1

                exp_avg = torch.mul(exp_avg, beta1) + (1 - beta1)*grad
                exp_avg_sq = torch.mul(exp_avg_sq, beta2) + (1-beta2)*(grad*grad)

                momentum = exp_avg/(1-beta1**state['step'])
                variance = exp_avg_sq/(1-beta2**state['step'])
                std_dev = torch.sqrt( exp_avg_sq + group['eps'] )

                p.data += group['lr'] * momentum / std_dev

                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq
                state['history'] = torch.zeros_like(p.data)


class AccumulatingET(Optimizer):
    def __init__(self, params, config):
        super().__init__(params, config)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(exp_avg=torch.zeros_like(p.data),
                                     exp_avg_sq=torch.zeros_like(p.data),
                                     trace=torch.zeros_like(p.data),
                                     step=0)

    def set(self):
        # NOTE: This zeros out the grads as well
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                grad, trace = p.grad.data, state['trace']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas'][0], group['betas'][1]
                state['step'] += 1
                exp_avg = torch.mul(exp_avg, beta1) + (1 - beta1)*grad
                exp_avg_sq = torch.mul(exp_avg_sq, beta2) + (1-beta2)*(grad*grad)
                momentum = exp_avg/(1-beta1**state['step'])
                variance = exp_avg_sq/(1-beta2**state['step'])
                std_dev = torch.sqrt( exp_avg_sq + group['eps'] )
                grad = momentum / std_dev
                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq

                trace = group['lam'] * group['gam'] * trace + grad
                if group['replacing']:
                    trace = torch.clamp(trace, -group['clip'], +group['clip'])
                p.grad.data = torch.zeros_like(p.data) # zero grad
                state['trace'] = trace

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trace'] = torch.zeros_like(p.data)

    def broadcast(self, residual):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                trace = state['trace']
                # print(f'Max {torch.max(trace)}, Min {torch.min(trace)}')
                p.data += group['lr'] * residual * trace


class ReplacingET(Optimizer):
    def __init__(self, params, config):
        super().__init__(params, config)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p] = dict(exp_avg=torch.zeros_like(p.data),
                                     exp_avg_sq=torch.zeros_like(p.data),
                                     trace=torch.zeros_like(p.data),
                                     step=0)

    def set(self):
        # NOTE: This zeros out the grads as well
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                grad, trace = p.grad.data, state['trace']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas'][0], group['betas'][1]
                state['step'] += 1
                exp_avg = torch.mul(exp_avg, beta1) + (1 - beta1)*grad
                exp_avg_sq = torch.mul(exp_avg_sq, beta2) + (1-beta2)*(grad*grad)
                momentum = exp_avg/(1-beta1**state['step'])
                variance = exp_avg_sq/(1-beta2**state['step'])
                std_dev = torch.sqrt( exp_avg_sq + group['eps'] )
                grad = momentum / std_dev
                state['exp_avg'], state['exp_avg_sq'] = exp_avg, exp_avg_sq

                trace = torch.where(torch.abs(grad) <= 0.1, group['lam'] * group['gam'] * trace, grad)
                p.grad.data = torch.zeros_like(p.data) # zero grad
                state['trace'] = trace

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['trace'] = torch.zeros_like(p.data)

    def broadcast(self, residual):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                trace = state['trace']
                # print(f'Max {torch.max(trace)}, Min {torch.min(trace)}')
                p.data += group['lr'] * residual * trace