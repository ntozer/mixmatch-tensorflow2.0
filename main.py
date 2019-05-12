import argparse

import tensorflow as tf

from preprocess import load_data


def get_args():
    parser = argparse.ArgumentParser('parameters')

    parser.add_argument('--dataset', type=str, default='SVHN',
                        help='dataset used for training (e.g. CIFAR10, CIFAR100, SVHN, SVHN+EXTRA)')
    parser.add_argument('--epochs', type=int, default=64, help="number of epochs, (default: 64)")
    parser.add_argument('--batch-size',  type=int, default=256, help='examples per batch (default: 256)')
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="learning_rate, (default: 0.01)")

    return parser.parse_args()


def main():
    args = get_args()
    train, test = load_data(args)
    print(args)


if __name__ == '__main__':
    main()