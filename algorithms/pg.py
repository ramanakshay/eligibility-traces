import torch
import numpy as np
from dotmap import DotMap


class BatchPolicyGradient(object):
    def __init__(self, env, model, config, logger):
        self.env = env
        self.model = model
        self.config = config
        self.logger = logger

    def episode_rollout(self):
        # rollout single episode
        max_episode_length = self.config.max_episode_length
    
        ep_obs, ep_acts, ep_rewards, ep_log_probs = [], [], [], []
    
        obs, _ = self.env.reset()
        for ep_step in range(max_episode_length):
            ep_obs.append(obs)
    
            action, dist = self.model.get_action(obs, False)
            log_prob = dist.log_prob(action)
            act, log_prob = action.detach().numpy()[0], log_prob.detach().item()
            obs, reward, terminated, truncated, info = self.env.step(act)
    
            ep_acts.append(act)
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
    
            if terminated or truncated:
                break
    
        ep_obs = torch.tensor(np.array(ep_obs))
        ep_acts = torch.tensor(np.array(ep_acts))
        ep_log_probs = torch.tensor(np.array(ep_log_probs))
        ep_rewards = torch.tensor(np.array(ep_rewards))
    
        return ep_obs, ep_acts, ep_log_probs, ep_rewards
    
    def batch_rollout(self):
        timesteps_per_batch = self.config.timesteps_per_batch
    
        # 2D tensor first index -> episode number
        batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_lens = [], [], [], [], []
    
        step = 0
        while (step < timesteps_per_batch):
            ep_obs, ep_acts, ep_log_probs, ep_rewards = self.episode_rollout()
            batch_obs.append(ep_obs)
            batch_acts.append(ep_acts)
            batch_log_probs.append(ep_log_probs)
            batch_rewards.append(ep_rewards)
            batch_lens.append(len(ep_obs))
            step += len(ep_obs)
    
        return batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_lens
    
    def learn(self, total_timesteps):
        iteration = 0
        timestep = 0
        while (timestep < total_timesteps):
            batch_obs, batch_acts, batch_log_probs, batch_rewards, batch_lens = self.batch_rollout()

            batch_sum = [ep_rewards.sum().item() for ep_rewards in batch_rewards]
            avg_ep_rew = sum(batch_sum)/len(batch_sum)
            avg_ep_lens = np.mean(batch_lens)

            iteration += 1
            timestep += np.sum(batch_lens)

            # calculate log_probs, and advantages
            actor_loss, critic_loss = self.evaluate(batch_obs, batch_acts, batch_rewards, batch_log_probs)

            self.model.reset()
            actor_loss.backward(retain_graph=True)
            if critic_loss is not None:
                critic_loss.backward()
            self.model.update()

            # update logger
            self.logger.update_data({
                'iteration': iteration,
                'timestep': timestep,
                'avg_ep_rew': avg_ep_rew,
                'avg_ep_lens': avg_ep_lens,
                'actor_loss': actor_loss.detach().item(),
            })

            self.logger.print_summary()

    def evaluate(self):
        raise NotImplementedError('`evaluate` function not implemented.')