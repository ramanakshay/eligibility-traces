from agent.model import DiscreteActorCriticv1, DiscreteActorCriticv2
from algorithm.train import ContinualPolicyGradientv1, ContinualPolicyGradientv2

from algorithm.test import Evaluator

import gymnasium as gym

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config : DictConfig) -> None:
    ## ENVIRONMENT ##
    env = gym.make(config.env.name, max_episode_steps = config.env.max_episode_steps)
    print('Environment Loaded.')

    ## MODEL ##
    agent = DiscreteActorCriticv2(config.agent)
    print('Agent Created.')

    ## ALGORITHM ##
    print('Running Algorithm.')
    # alg = Evaluator(agent, env, config)
    # alg.render()
    alg = ContinualPolicyGradientv2(agent, env, config.algorithm)
    alg.run()


if __name__ == "__main__":
    main()