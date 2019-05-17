import argparse
import os

import tensorflow as tf
import yaml

from preprocess import load_data


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--seed', type=int, default=None, help='seed for repeatable results')
    parser.add_argument('--dataset', type=str, default='svhn',
                        help='dataset used for training (e.g. cifar10, cifar100, svhn, svhn+extra)')
    parser.add_argument('--epochs', type=int, default=1024, help='number of epochs, (default: 64)')
    parser.add_argument('--batch-size',  type=int, default=64, help='examples per batch (default: 256)')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='learning_rate, (default: 0.01)')
    parser.add_argument('--labelled-examples', type=int, default=4000, help='number labelled examples (default: 4000')
    parser.add_argument('--T', type=float, default=0.5, help='temperature sharpening ratio (default: 0.5)')
    parser.add_argument('--K', type=int, default=2, help='number of rounds of augmentation (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='param for sampling from Beta distribution (default: 0.75)')
    parser.add_argument('--lambdaU', type=int, default=100, help='multiplier for unlabelled loss (default: 100)')

    parser.add_argument('--config-path', type=str, default=None, help='path to yaml config file, overwrites args')

    return parser.parse_args()


def load_config(args):
    with open(args['config_path'], 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args


def main():
    args = vars(get_args())
    if os.path.exists(args['config_path']):
        args = load_config(args)

    trainX, trainU, test = load_data(args)
    print(args)


if __name__ == '__main__':
    main()