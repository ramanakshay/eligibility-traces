from agent.gae_models import DiscreteActorCritic
from algorithms.gae import GAE
import gymnasium as gym
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
env = gym.make('Acrobot-v1')
env_name = env.unwrapped.spec.id
logger.update_info(f"Environment = {env_name}\n")
print(env_name)
print()

# model
model_config = DotMap({
    'obs_dim': env.observation_space.shape[0],
    'act_dim': env.action_space.n,
    'hidden_dim': 64,
    'lr': 5e-3,
    'gam': 0.99,
    'lam': 0.9
})
model = DiscreteActorCritic(model_config)
logger.update_info(str(model))
logger.update_info(str(model_config)+"\n")
print(model)
print(model_config)

# algorithm
alg_config = DotMap({
    'max_episode_length': 1000,
    'timesteps_per_update': 2500,
    'gamma': 0.99
})
alg = GAE(env, model, alg_config, logger)
total_timesteps = 1000000
alg.learn(total_timesteps)
logger.update_info("Algorithm: ")
logger.update_info("GAE")
logger.update_info(str(alg_config))
logger.update_info(f"Total Timesteps={total_timesteps}")

log_name = env_name + "_ContinualGAE"
logger.save(log_name)












