import torch
import numpy as np

class RolloutPolicyGradient(object):
    def __init__(self, env, model, config):
        self.env = env
        self.model = model
        self.config = config
        self.logger = {}

    def log_summary(self):
        timestep = self.logger['timestep']
        iteration = self.logger['iteration']
        actor_loss = self.logger['actor_loss'].item()
    
        batch_reward = self.logger['batch_rewards']
        batch_sum = [ep_rewards.sum().item() for ep_rewards in batch_reward]
        avg_ep_rew = sum(batch_sum)/len(batch_sum)
    
    
        avg_ep_rew = str(round(avg_ep_rew, 2))
        actor_loss = str(round(actor_loss, 5))
    
        print(flush=True)
        print(f"-------------------- Iteration #{iteration} --------------------", flush=True)
        # print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rew}", flush=True)
        print(f"Actor Loss: {actor_loss}", flush=True)
        print(f"Timesteps So Far: {timestep}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)
    
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
            self.logger['batch_rewards'] = batch_rewards

            iteration += 1
            self.logger['iteration'] = iteration

            timestep += np.sum(batch_lens)
            self.logger['timestep'] = timestep

            # calculate log_probs, and advantages
            actor_loss, critic_loss = self.evaluate(batch_obs, batch_acts, batch_rewards, batch_log_probs)

            self.model.reset()
            actor_loss.backward(retain_graph=True)
            if critic_loss is not None:
                critic_loss.backward()
            self.model.update()

            self.logger['actor_loss'] = actor_loss.detach()

            self.log_summary()
            
    def evaluate(self):
        raise NotImplementedError('`evaluate` function not implemented.')