import torch
from torch import nn
import numpy as np
from algorithms.pg import BatchPolicyGradient


class RTG(BatchPolicyGradient):
    def __init__(self, env, model, config, logger):
        BatchPolicyGradient.__init__(self, env, model, config, logger)

    def compute_rtgs(self, batch_rewards):
        gamma = self.config.gamma
        batch_rtgs = []

        for ep_rewards in batch_rewards:
            ep_rtgs = []
            total_reward = 0
            for rew in reversed(ep_rewards):
                total_reward = rew + gamma * total_reward
                ep_rtgs.insert(0, total_reward)
            ep_rtgs = torch.tensor(np.array(ep_rtgs))
            batch_rtgs.append(ep_rtgs)

        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts, batch_rewards, batch_log_probs):
        # calculate advantages
        batch_rtgs = self.compute_rtgs(batch_rewards)
        batch_rtgs = torch.cat(batch_rtgs)
        normalized_rtgs = (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-10)
        advantages = normalized_rtgs

        # calculate log_probs
        batch_obs, batch_acts = torch.cat(batch_obs), torch.cat(batch_acts)
        _, dist = self.model.get_action(batch_obs)
        log_probs = dist.log_prob(batch_acts)

        actor_loss = (-advantages * log_probs).mean()
        actor_loss.backward(retain_graph=True)
        return actor_loss, None


class BaselineRTG(BatchPolicyGradient):
    def __init__(self, env, model, config, logger):
            BatchPolicyGradient.__init__(self, env, model, config, logger)

    def compute_rtgs(self, batch_rewards):
        gamma = self.config.gamma
        batch_rtgs = []

        for ep_rewards in batch_rewards:
            ep_rtgs = []
            total_reward = 0
            for rew in reversed(ep_rewards):
                total_reward = rew + gamma * total_reward
                ep_rtgs.insert(0, total_reward)
            ep_rtgs = torch.tensor(np.array(ep_rtgs))
            batch_rtgs.append(ep_rtgs)

        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts, batch_rewards, batch_log_probs):
        # calculate advantages
        batch_rtgs = self.compute_rtgs(batch_rewards)
        batch_rtgs = torch.cat(batch_rtgs)

        batch_obs, batch_acts = torch.cat(batch_obs), torch.cat(batch_acts)

        values = self.model.get_value(batch_obs).squeeze()
        advantages = batch_rtgs - values.clone().detach()
        normalized_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        _, dist = self.model.get_action(batch_obs)
        log_probs = dist.log_prob(batch_acts)
        actor_loss = (-normalized_adv * log_probs).mean()
        critic_loss = nn.MSELoss()(batch_rtgs.float(), values)

        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        return actor_loss, critic_loss


class TDResidual(BatchPolicyGradient):
    def __init__(self, env, model, config, logger):
        BatchPolicyGradient.__init__(self, env, model, config, logger)

    def compute_residuals(self, batch_obs, batch_rewards):
        gamma = self.config.gamma
        batch_res = []

        for ep_num in range(len(batch_rewards)):
            ep_obs = batch_obs[ep_num]
            ep_rew = batch_rewards[ep_num]
            ep_values = self.model.get_value(ep_obs, False).squeeze()
            ep_values_prime = torch.cat((ep_values[1:], torch.tensor([0.0])))
            ep_res = ep_rew + gamma * ep_values_prime - ep_values
            batch_res.append(ep_res)

        return batch_res

    def evaluate(self, batch_obs, batch_acts, batch_rewards, batch_log_probs):
        # calculate advantages
        batch_res = self.compute_residuals(batch_obs, batch_rewards)
        batch_obs, batch_acts = torch.cat(batch_obs), torch.cat(batch_acts)

        residuals = torch.cat(batch_res)
        values = self.model.get_value(batch_obs).squeeze()
        advantages = residuals
        normalized_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        _, dist = self.model.get_action(batch_obs)
        log_probs = dist.log_prob(batch_acts)
        actor_loss = (-normalized_adv * log_probs).mean()
        critic_loss = (-normalized_adv * values).mean()

        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

        return actor_loss, critic_loss


class GAE(BatchPolicyGradient):
    def __init__(self, env, model, config, logger):
        BatchPolicyGradient.__init__(self, env, model, config, logger)

    def compute_lamda_returns(self, batch_obs, batch_rewards):
        gamma = self.config.gamma
        lam = self.config.lam

        batch_lam = []

        for ep_num in range(len(batch_rewards)):
            ep_obs = batch_obs[ep_num]
            ep_rew = batch_rewards[ep_num]
            ep_values = self.model.get_value(ep_obs, False).squeeze()
            ep_values_prime = torch.cat((ep_values[1:], torch.tensor([0.0])))
            ep_res = ep_rew + gamma * ep_values_prime - ep_values

            ep_lam = []
            residual_sum = 0
            for res in reversed(ep_res):
                residual_sum = res + lam * gamma * residual_sum
                ep_lam.insert(0, residual_sum)
            ep_lam = torch.tensor(np.array(ep_lam))
            batch_lam.append(ep_lam)

        return batch_lam

    def evaluate(self, batch_obs, batch_acts, batch_rewards, batch_log_probs):
        # calculate advantages
        batch_lam = self.compute_lamda_returns(batch_obs, batch_rewards)
        batch_obs, batch_acts = torch.cat(batch_obs), torch.cat(batch_acts)

        batch_lam = torch.cat(batch_lam)
        values = self.model.get_value(batch_obs).squeeze()
        advantages = batch_lam
        # print(advantages.min(), advantages.max())
        normalized_adv = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        # print(normalized_adv.min(), normalized_adv.max())

        _, dist = self.model.get_action(batch_obs)
        log_probs = dist.log_prob(batch_acts)
        actor_loss = (-normalized_adv * log_probs).mean()
        critic_loss = (-normalized_adv * values).mean()

        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

        return actor_loss, critic_loss