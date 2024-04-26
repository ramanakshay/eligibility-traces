from agent.models import ContinuousActor
from algorithms.vpg import RTG, BaselineRTG
import gymnasium as gym
from dotmap import DotMap

# environment
env = gym.make('Pendulum-v1')
env_name = env.unwrapped.spec.id
print(env_name)
print()

# model
model_config = DotMap({
    'obs_dim': env.observation_space.shape[0],
    'act_dim': env.action_space.shape[0],
    'hidden_dim': 64,
    'lr': 3e-4
})
model = ContinuousActor(model_config)
print(model_config)
print(model)


# algorithm
alg_config = DotMap({
    'max_episode_length': 200,
    'timesteps_per_batch': 2048,
    'gamma': 0.99
})
alg = RTG(env, model, alg_config)
total_timesteps = 2005000
alg.learn(total_timesteps)





