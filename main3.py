from agent.models import DiscreteActorCritic, DiscreteActor
from algorithms.vpg import BaselineRTG, RTG, TDR, GAE
import gymnasium as gym
import particle_envs
from logger import Logger
from dotmap import DotMap
import torch

# logger
logger = Logger()

# seed
seed = None
if seed != None:
    assert(type(seed) == int)
    torch.manual_seed(seed)
    print(f"Successfully set seed to {seed}")
    logger.update_info(f"Seed = {seed}")


# environment
env = gym.make('particle-v0')
env_name = env.unwrapped.spec.id
logger.update_info(f"Environment = {env_name}\n")
print(env_name)

# model
model_config = DotMap({
    'obs_dim': env.observation_space.shape[0],
    'act_dim': env.action_space.shape[0],
    'hidden_dim': 64,
    'lr': 1e-3
})
model = DiscreteActor(model_config)
logger.update_info(str(model))
logger.update_info(str(model_config)+"\n")
print(model)
print(model_config)

# algorithm
alg_config = DotMap({
    'max_episode_length': 100,
    'timesteps_per_batch': 2048,
    'gamma': 0.99,
    'lam': 0.9
})
alg = RTG(env, model, alg_config, logger)
total_timesteps = 1000000
alg.learn(total_timesteps)
logger.update_info("Algorithm: ")
logger.update_info("GAE")
logger.update_info(str(alg_config))
logger.update_info(f"Total Timesteps={total_timesteps}")

log_name = env_name + "_GAE"
logger.save(log_name)












