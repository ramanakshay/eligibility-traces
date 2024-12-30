import numpy as np
import torch
from tqdm import trange


class ContinualPolicyGradientv1(object):
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config

    def run_episode(self):
        self.agent.reset()
        ep_rew, ep_len = 0, 0
        obs, _ = self.env.reset()
        obs = torch.tensor(np.asarray(obs))
        while True:
            act, dist = self.agent.get_action(obs, grad=False)
            act = act.detach().numpy()[0] # move to cpu
            next_obs, reward, terminated, truncated, info = self.env.step(act)
            ep_rew += reward
            ep_len += 1

            obs = torch.tensor(np.asarray(obs))
            act = torch.tensor(np.asarray(act))
            next_obs = torch.tensor(np.asarray(next_obs))
            self.agent.learn(obs, act, reward, next_obs)
            obs = next_obs

            if terminated or truncated:
                break

        return ep_rew, ep_len

    def run(self):
        total_episodes = self.config.total_episodes
        episodes_per_update = self.config.episodes_per_update
        iteration = 0
        avg_ep_rew, avg_ep_len = 0, 0

        for episode in range(total_episodes):
            ep_rew, ep_len = self.run_episode()
            avg_ep_rew += ep_rew
            avg_ep_len += ep_len

            if episode%episodes_per_update == 0:
                avg_ep_rew /= episodes_per_update
                avg_ep_len /= episodes_per_update
                print(f'Episodes {episode}')
                print(f'Average Episode Reward: {avg_ep_rew}')
                print(f'Average Episode Length: {avg_ep_len}')
                avg_ep_rew, avg_ep_len = 0, 0

        self.agent.save_weights()



class ContinualPolicyGradientv2(object):
    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config

    def run_episode(self):
        self.agent.reset()
        ep_rew, ep_len = 0, 0
        obs, _ = self.env.reset()
        obs = torch.tensor(np.asarray(obs))
        while True:
            act = self.agent.act(obs)
            act = act.detach().numpy()[0] # move to cpu

            obs, reward, terminated, truncated, info = self.env.step(act)
            obs = torch.tensor(np.asarray(obs))
            self.agent.learn(obs, reward)

            ep_rew += reward
            ep_len += 1
            if terminated or truncated:
                break

        return ep_rew, ep_len

    def run(self):
        total_episodes = self.config.total_episodes
        episodes_per_update = self.config.episodes_per_update
        iteration = 0
        avg_ep_rew, avg_ep_len = 0, 0
        pbar = trange(total_episodes)
        for episode in pbar:
            ep_rew, ep_len = self.run_episode()
            avg_ep_rew += ep_rew
            avg_ep_len += ep_len

            pbar.set_description(f'Episode {episode}')
            pbar.set_postfix(reward=ep_rew, length=ep_len)

            if episode%episodes_per_update == 0:
                avg_ep_rew /= episodes_per_update
                avg_ep_len /= episodes_per_update
                # print(f'Episodes {episode}')
                # print(f'Average Episode Reward: {avg_ep_rew}')
                # print(f'Average Episode Length: {avg_ep_len}')
                avg_ep_rew, avg_ep_len = 0, 0

        self.agent.save_weights()





