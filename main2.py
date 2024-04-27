from agent.models import ContinuousActor
from algorithms.vpg import RTG, BaselineRTG
import gymnasium as gym
from dotmap import DotMap
from logger import Logger
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
env = gym.make('LunarLanderContinuous-v2')
env_name = env.unwrapped.spec.id
logger.update_info(f"Environment = {env_name}\n")
print(env_name)
print()

# model
model_config = DotMap({
    'obs_dim': env.observation_space.shape[0],
    'act_dim': env.action_space.shape[0],
    'hidden_dim': 64,
    'lr': 2.5e-4
})
model = ContinuousActor(model_config)
logger.update_info(str(model))
logger.update_info(str(model_config)+"\n")
print(model)
print(model_config)

# algorithm
alg_config = DotMap({
    'max_episode_length': 1000,
    'timesteps_per_batch': 1024,
    'gamma': 0.999
})
alg = RTG(env, model, alg_config, logger)
total_timesteps = 3002500
alg.learn(total_timesteps)
logger.update_info("Algorithm: ")
logger.update_info("RTG")
logger.update_info(str(alg_config))
logger.update_info(f"Total Timesteps={total_timesteps}")

log_name = env_name + "_RTG"
logger.save(log_name)





