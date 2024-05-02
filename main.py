from dotmap import DotMap
from run import train
import argparse


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    parser.add_argument('--env', dest='env', type=str, default='Acrobot-v1')
    parser.add_argument('--model', dest='model', type=str, default='discrete-actor-critic')
    parser.add_argument('--alg', dest='alg', type=str, default='gae')
    args = parser.parse_args()

    SEED = [40, 13, 42]
    for seed in SEED:
        args.seed = seed
        train(args)
    # eval(args)