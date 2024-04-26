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
        return actor_loss, critic_loss