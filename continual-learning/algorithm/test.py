import numpy as np
import torch
from tqdm import trange
import cv2
import gymnasium as gym


class Evaluator(object):
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
            obs, reward, terminated, truncated, info = self.env.step(act)
            obs = torch.tensor(np.asarray(obs))
            ep_rew += reward
            ep_len += 1

            if terminated or truncated:
                break

        return ep_rew, ep_len

    def run(self):
        total_episodes = self.config.algorithm.total_episodes
        avg_ep_rew, avg_ep_len = 0, 0
        pbar = trange(total_episodes)
        for episode in pbar:
            ep_rew, ep_len = self.run_episode()

            pbar.set_description(f'Episode {episode}')
            pbar.set_postfix(reward=ep_rew, length=ep_len)

            avg_ep_rew += ep_rew
            avg_ep_len += ep_len

        avg_ep_rew /= total_episodes
        avg_ep_len /= total_episodes

        print(f"-------------------- EVALUATION --------------------")
        print(f"Performance over {total_episodes} epsidoes")
        print(f"Average Episodic Length: {round(avg_ep_len, 2)}")
        print(f"Average Episodic Return: {round(avg_ep_rew, 2)}")
        print(f"------------------------------------------------------")
        print(f"")

    def render(self):
        env = gym.make(self.config.env.name, max_episode_steps = self.config.env.max_episode_steps,
                       render_mode='human')
        num_episodes = 5
        for ep in range(num_episodes):
            obs, _ = env.reset()
            obs = torch.tensor(np.asarray(obs))
            count = 0
            while True:
                act, dist = self.agent.get_action(obs, False)
                act = act.detach().numpy()[0]
                obs, reward, terminated, truncated, info = env.step(act)
                obs = torch.tensor(np.asarray(obs))
                count += 1
                if terminated or truncated:
                    print(count)
                    break






