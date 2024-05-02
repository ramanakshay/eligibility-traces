from agent.models import ContinuousActorCritic, ContinuousActor
from algorithms.vpg import BaselineRTG, RTG, TDR, GAE
from agent.networks import LargeMLP
import gymnasium as gym
import particle_envs
from logger import Logger
from dotmap import DotMap
import torch
from evaluate import Evaluator

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
env_config = DotMap({
    'height': 90,
    'width': 90,
    'step_size': 7,
    'block': [[0, 65, 15, 30], [25, 89, 55, 70]],
    'start': [0,10,0,10],
    'goal': [79,89,31,54],
    'reward_type': None,
})
        
env = gym.make('particle-v0', **env_config.toDict())
env_name = env.unwrapped.spec.id
logger.update_info(f"Environment = {env_name}\n")
print(env_name)

# model
model_config = DotMap({
    'obs_dim': env.observation_space.shape[0],
    'act_dim': env.action_space.shape[0],
    'hidden_dim': 256,
    'lr': 5e-4
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
    'lam': 0.9
})


alg = GAE(env, model, alg_config, logger)
total_timesteps = 50000

# task 1
alg.learn(total_timesteps)

evaluator = Evaluator(env, model)
evaluator.results()
evaluator.render()
# 
# task 2
env_config = DotMap({
    'height': 90,
    'width': 90,
    'step_size': 7,
    'reward_type': None,
    'block': [[0, 65, 15, 30], [25, 89, 55, 70]],
    'start': [79,89,31,54],
    'goal': [40,50,31,54],
})
        
env = gym.make('particle-v0', **env_config.toDict())
env_name = env.unwrapped.spec.id
logger.update_info(f"Environment = {env_name}\n")
print(env_name)
# # 
alg.env = env
model.reset_critic()
total_timesteps = 50000
alg.learn(total_timesteps)

evaluator = Evaluator(env, model)
evaluator.results()
evaluator.render()



# task 3
env_config = DotMap({
    'height': 90,
    'width': 90,
    'step_size': 7,
    'block': [[0, 65, 15, 30], [25, 89, 55, 70]],
    'start': [40,50,31,54],
})
        
env = gym.make('particle-v0', **env_config.toDict())
env_name = env.unwrapped.spec.id
logger.update_info(f"Environment = {env_name}\n")
print(env_name)
# # 
alg.env = env
model.reset_critic()
total_timesteps = 50000
alg.learn(total_timesteps)

evaluator = Evaluator(env, model)
evaluator.results()
evaluator.render()




# final task
env_config = DotMap({
    'height': 90,
    'width': 90,
    'step_size': 7,
    'block': [[0, 65, 15, 30], [25, 89, 55, 70]],
    'start': [0,10,0,10],
})
env = gym.make('particle-v0', **env_config.toDict())
env_name = env.unwrapped.spec.id
logger.update_info(f"Environment = {env_name}\n")
print(env_name)

total_timesteps = 50000
alg.env = env
model.reset_critic()
alg.learn(total_timesteps)

evaluator = Evaluator(env, model)
evaluator.results()
evaluator.render()


# task 2
# logger.update_info("Algorithm: ")
# logger.update_info("GAE")
# logger.update_info(str(alg_config))
# logger.update_info(f"Total Timesteps={total_timesteps}")
# # log_name = env_name + "_GAE"
# # logger.save(log_name)
# 
# # Evaluator
# evaluator = Evaluator(env, model)
# evaluator.results()
# evaluator.render()












