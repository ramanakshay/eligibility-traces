import torch
from torch import nn
import numpy as np
from algorithms.pg import ContinualPolicyGradient
from dotmap import DotMap


class GAE(ContinualPolicyGradient):
   def __init__(self, env, model, config, logger):
      ContinualPolicyGradient.__init__(self, env, model, config, logger)
   
   def evaluate(self, obs, act, reward, next_obs):
      gamma = self.config.gamma
      
      # calculate td residual
      value = self.model.get_value(obs)
      next_value = self.model.get_value(next_obs, False)
      residual = reward + gamma * next_value - value.clone().detach()
      
      
      # calculate log prob
      _, dist = self.model.get_action(obs)
      log_prob = dist.log_prob(act)
      
      # calculate gradients
      actor_grad = log_prob.backward()
      critic_grad = value.backward()
      
      self.model.trace()
      self.model.broadcast(residual.item())

         