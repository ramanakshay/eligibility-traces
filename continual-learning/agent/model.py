import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent.network import MLP
from agent.optimizer import AccumulatingET, ReplacingET

class DiscreteActorCriticv1(object):
    def __init__(self, config):
        self.config = config
        obs_dim = self.config.obs_dim
        hidden_dim = self.config.hidden_dim
        act_dim = self.config.act_dim

        self.actor = MLP(obs_dim, hidden_dim, act_dim)
        self.critic = MLP(obs_dim, hidden_dim, 1)

        self.optimizer = AccumulatingET([
                {'params': self.actor.parameters()},
                {'params': self.critic.parameters()}
            ], self.config.optim)

        self.weights_path = config.weights_path
        if config.from_pretrained:
            self.load_weights()

    def __repr__(self):
        return f"Discrete Actor Critic Model: \n\
    Actor Network = {type(self.actor).__name__} \n\
    Critic Network = {type(self.critic).__name__} \n\
    Optimizer = {type(self.optimizer).__name__}"

    def save_weights(self):
        torch.save(self.actor.state_dict(),  f'{self.weights_path}discrete_actor.pth')
        torch.save(self.critic.state_dict(),  f'{self.weights_path}discrete_critic.pth')

    def load_weights(self):
        self.actor.load_state_dict(torch.load(f"{self.weights_path}discrete_actor.pth", weights_only=True))
        self.critic.load_state_dict(torch.load(f"{self.weights_path}discrete_critic.pth", weights_only=True))

    def set(self):
        self.optimizer.set()

    def broadcast(self, residual):
        self.optimizer.broadcast(residual)

    def reset(self):
        self.optimizer.reset()

    def step(self):
        self.optimizer.step()

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

    def learn(self, obs, act, reward, next_obs):
        gam = self.config.optim.gam

        # calculate td residual
        value = self.get_value(obs, grad=True)
        next_value = self.get_value(next_obs, grad=False)
        residual = reward + gam * next_value - value.clone().detach()

        # calculate log prob
        _, dist = self.get_action(obs, grad=True)
        log_prob = dist.log_prob(act)

        # calculate gradients
        actor_grad = log_prob.backward()
        critic_grad = value.backward()

        self.set()
        self.broadcast(residual.item())
        # self.step()


class DiscreteActorCriticv2(object):
    def __init__(self, config):
        self.config = config
        obs_dim = self.config.obs_dim
        hidden_dim = self.config.hidden_dim
        act_dim = self.config.act_dim

        self.actor = MLP(obs_dim, hidden_dim, act_dim)
        self.critic = MLP(obs_dim, hidden_dim, 1)

        self.optimizer = AccumulatingET([
            {'params': self.actor.parameters()},
            {'params': self.critic.parameters()}
        ], self.config.optim)

        self.weights_path = config.weights_path
        if config.from_pretrained:
            self.load_weights()


    def __repr__(self):
        return f"Discrete Actor Critic Model: \n\
            Actor Network = {type(self.actor).__name__} \n\
            Critic Network = {type(self.critic).__name__} \n\
            Optimizer = {type(self.optimizer).__name__}"

    def save_weights(self):
        torch.save(self.actor.state_dict(),  f'{self.weights_path}discrete_actor.pth')
        torch.save(self.critic.state_dict(),  f'{self.weights_path}discrete_critic.pth')

    def load_weights(self):
        self.actor.load_state_dict(torch.load(f"{self.weights_path}discrete_actor.pth", weights_only=True))
        self.critic.load_state_dict(torch.load(f"{self.weights_path}discrete_critic.pth", weights_only=True))

    def set(self):
        self.optimizer.set()

    def broadcast(self, residual):
        self.optimizer.broadcast(residual)

    def reset(self):
        self.optimizer.reset()

    def step(self):
        self.optimizer.step()

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

    def act(self, obs):
        # calculate td residual
        self.obs = obs
        self.value = self.get_value(self.obs, grad=True)

        # calculate log prob
        action, dist = self.get_action(self.obs, grad=True)
        self.log_prob = dist.log_prob(action)

        return action

    def learn(self, next_obs, reward):
        gam = self.config.optim.gam

        # calculate td-error
        next_value = self.get_value(next_obs, grad=False).clone().detach()
        value = self.value.clone().detach()
        residual = reward + gam * next_value - value

        # calculate gradients
        actor_grad = self.log_prob.backward()
        critic_grad = self.value.backward()

        # set trace and broadcast
        self.set()
        self.broadcast(residual.item())