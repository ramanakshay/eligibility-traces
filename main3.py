from agent.models import DiscreteActorCritic, DiscreteActor
from algorithms.vpg import BaselineRTG, RTG
import gymnasium as gym
from dotmap import DotMap

# environment
env = gym.make('Acrobot-v1')
env_name = env.unwrapped.spec.id
print(env_name)
print()

# model
model_config = DotMap({
    'obs_dim': env.observation_space.shape[0],
    'act_dim': env.action_space.n,
    'hidden_dim': 64,
    'lr': 3e-4
})

model = DiscreteActorCritic(model_config)
print(model_config)
print(model)

alg_config = DotMap({
    'max_episode_length': 1000,
    'timesteps_per_batch': 2048,
    'gamma': 0.99
})
alg = BaselineRTG(env, model, alg_config)
total_timesteps = 1000000
alg.learn(total_timesteps)





