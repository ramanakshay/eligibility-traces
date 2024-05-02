from agent.models import ContinuousActorCritic
from algorithms.vpg import GAE
import gymnasium as gym
from dotmap import DotMap
from logger import Logger
import torch
import particle_envs

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
print()

# model
# model
model_config = DotMap({
    'obs_dim': env.observation_space.shape[0],
    'act_dim': env.action_space.shape[0],
    'hidden_dim': 64,
    'lr': 2e-4
})
model = ContinuousActorCritic(model_config)
logger.update_info(str(model))
logger.update_info(str(model_config)+"\n")
print(model)
print(model_config)

# algorithm
alg_config = DotMap({
    'max_episode_length': 200,
    'timesteps_per_batch': 1000,
    'gamma': 1.0,
    'lam': 0.9,
})
alg = GAE(env, model, alg_config, logger)
total_timesteps = 50000
alg.learn(total_timesteps)
logger.update_info("Algorithm: ")
logger.update_info("Continual GAE")
logger.update_info(str(alg_config))
logger.update_info(f"Total Timesteps={total_timesteps}")

log_name = env_name + "_ContGAE"
logger.save(log_name)





