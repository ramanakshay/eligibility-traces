from agent.networks import MLP
from agent.optimizers import ETSGD, ETAdam

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical


class ContinuousActorCritic(object):
    def __init__(self, config):
        self.config = config
        self.obs_dim = self.config.obs_dim
        self.act_dim = self.config.act_dim

        self.actor = MLP(self.obs_dim, self.act_dim, self.config.hidden_dim)
        self.critic = MLP(self.obs_dim, 1, self.config.hidden_dim)
        
        self.lr = self.config.lr
        self.gam = self.config.gam
        self.lam = self.config.lam
        self.clip = self.config.clip
        self.replacing = self.config.replacing
        
        self.actor_optim = ETAdam(self.actor.parameters(), lr = self.lr, gam=self.gam, lam = self.lam,
                                   clip=self.clip, replacing=self.replacing)
        self.critic_optim = ETAdam(self.critic.parameters(), lr = self.lr, gam=self.gam, lam = self.lam,
                                    clip=self.clip, replacing=self.replacing)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def __repr__(self):
        return f"Continuous Actor Critic Model: \n\
            Actor Network = {type(self.actor).__name__} \n\
            Actor Optimizer = {type(self.actor_optim).__name__} \n\
            Critic Network = {type(self.critic).__name__} \n\
            Critic Optimizer = {type(self.critic_optim).__name__}"
    
    def trace(self):
        self.actor_optim.set_trace()
        self.critic_optim.set_trace()
        
    def broadcast(self, residual):
        self.actor_optim.broadcast(residual)
        self.critic_optim.broadcast(residual)

    def reset(self):
        # reset trace
        self.actor_optim.reset_trace()
        self.critic_optim.reset_trace()

    def update(self):
        self.actor_optim.step()
        self.critic_optim.step()
    
    def get_action(self, obs, grad=True):
        with torch.set_grad_enabled(grad):
            mean = self.actor(obs)
            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
        return action, dist

    def get_value(self, obs, grad=True):
        with torch.set_grad_enabled(grad):
            value = self.critic(obs)
        return value


class DiscreteActorCritic(object):
    def __init__(self, config):
        self.config = config
        self.obs_dim = self.config.obs_dim
        self.act_dim = self.config.act_dim

        self.actor = MLP(self.obs_dim, self.act_dim, self.config.hidden_dim)
        self.critic = MLP(self.obs_dim, 1, self.config.hidden_dim)

        self.lr = self.config.lr
        self.gam = self.config.gam
        self.lam = self.config.lam
        self.clip = self.config.clip
        self.replacing = self.config.replacing

        # NOTE: ETSGD Maximizes
        self.actor_optim = ETAdam(self.actor.parameters(), lr = self.lr, gam=self.gam, lam = self.lam,
                    clip=self.clip, replacing=self.replacing)
        self.critic_optim = ETAdam(self.critic.parameters(), lr = self.lr, gam=self.gam, lam = self.lam,
                    clip=self.clip, replacing=self.replacing)


    def __repr__(self):
        return f"Discrete Actor Critic Model: \n\
    Actor Network = {type(self.actor).__name__} \n\
    Actor Optimizer = {type(self.actor_optim).__name__} \n\
    Critic Network = {type(self.critic).__name__} \n\
    Critic Optimizer = {type(self.critic_optim).__name__}"
    
    def trace(self):
            self.actor_optim.set_trace()
            self.critic_optim.set_trace()
            
    def broadcast(self, residual):
        self.actor_optim.broadcast(residual)
        self.critic_optim.broadcast(residual)
    
    def reset(self):
        # reset trace
        self.actor_optim.reset_trace()
        self.critic_optim.reset_trace()

    def update(self):
        self.actor_optim.step()
        self.critic_optim.step()

    def get_action(self, obs, grad=True):
        with torch.set_grad_enabled(grad):
            output = self.actor(obs)
            probs = F.softmax(output, dim=1)
            dist = Categorical(probs)
            action = dist.sample()
            return action, dist

    def get_value(self, obs, grad=True):
        with torch.set_grad_enabled(grad):
            value = self.critic(obs)
        return value