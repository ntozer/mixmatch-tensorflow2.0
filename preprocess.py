import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(args):
    train = None
    test = None
    if args.dataset == 'SVHN':
        train = tfds.load(name='svhn_cropped', split=tfds.Split.TRAIN)
        test = tfds.load(name='svhn_cropped', split=tfds.Split.TEST)
    elif args.dataset == 'SVHN+EXTRA':
        train = tfds.load(name='svhn_cropped', split=tfds.Split.TRAIN)
        extra = tfds.load(name='svhn_cropped', split=tfds.Split.EXTRA)
        test = tfds.load(name='svhn_cropped', split=tfds.Split.TEST)
    elif args.dataset == 'CIFAR10':
        train = tfds.load(name='cifar10', split=tfds.Split.TRAIN)
        test = tfds.load(name='cifar10', split=tfds.Split.TEST)
    elif args.dataset == 'CIFAR100':
        train = tfds.load(name='cifar100', split=tfds.Split.TRAIN)
        test = tfds.load(name='cifar100', split=tfds.Split.TEST)
    return train, test