import gymnasium as gym
import cv2

class Evaluator(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model
        self.model.reset()
    
    def results(self):
        num_episodes = 10
        avg_ep_rew, avg_ep_len = 0, 0
        for ep in range(num_episodes):
            ep_rew, ep_len = 0, 0
            obs, _ = self.env.reset()
            done = False
            while not done:
                ep_len += 1
    
                act, dist = self.model.get_action(obs, False)
                act = act.detach().numpy()[0]
                next_obs, reward, terminated, truncated, info = self.env.step(act)
                ep_rew += reward
                done = terminated or truncated                                                   

            avg_ep_rew += ep_rew
            avg_ep_len += ep_len
            
            
        avg_ep_rew = avg_ep_rew/num_episodes
        avg_ep_len = avg_ep_len/num_episodes
            
        print(f"-------------------- EVALUATION --------------------", flush=True)
        print(f"Performance over {num_episodes} epsidoes", flush=True)
        print(f"Average Episodic Length: {round(avg_ep_len, 2)}", flush=True)
        print(f"Average Episodic Return: {round(avg_ep_rew, 2)}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(f"")
        
    def render(self):
        # env = gym.make(self.env.unwrapped.spec.id, render_mode='human')
        num_episodes = 5
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            done = False

            img = self.env.render()
            cv2.imshow("Render", img)
            cv2.waitKey(25)

            while not done:
                act, dist = self.model.get_action(obs, False)
                act = act.detach().numpy()[0]
                next_obs, reward, terminated, truncated, info = self.env.step(act)
                done = terminated or truncated

                img = self.env.render()
                cv2.imshow("Render", img)
                cv2.waitKey(25)

        