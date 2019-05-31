import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(args):
    train = None
    test = None
    if args['dataset'] == 'svhn':
        dataset = tfds.load(name='svhn_cropped')
        train = convert_to_numpy(dataset['train'])
        test = convert_to_numpy(dataset['test'])
        num_labels = 10

    elif args['dataset'] == 'svhn+extra':
        dataset = tfds.load(name='svhn_cropped')
        train = convert_to_numpy(dataset['train'])
        extra = convert_to_numpy(dataset['extra'])
        for key in train.keys():
            train[key] = np.concatenate([train[key], extra[key]], axis=0)
        test = convert_to_numpy(dataset['test'])
        num_labels = 10

    elif args['dataset'] == 'cifar10':
        dataset = tfds.load(name='cifar10')
        train = convert_to_numpy(dataset['train'])
        test = convert_to_numpy(dataset['test'])
        num_labels = 10

    elif args['dataset'] == 'cifar100':
        dataset = tfds.load(name='cifar100')
        train = convert_to_numpy(dataset['train'])
        test = convert_to_numpy(dataset['test'])
        num_labels = 100

    # train = change_range(train, start=(0., 255.), end=(0., 1.))
    # test = change_range(test, start=(0., 255.), end=(0., 1.))
    train = convert_to_one_hot(train, num_labels)
    test = convert_to_one_hot(test, num_labels)
    for key in test.keys():
        test[key] = test[key].astype(dtype=np.float32)
    trainX, trainU = generate_labelled_and_unlabelled_datasets(args, train)

    return trainX, trainU, test, num_labels


def convert_to_numpy(dataset):
    np_generator = tfds.as_numpy(dataset)
    examples = {}
    for example in np_generator:
        for key in example:
            try:
                examples[key].append(example[key])
            except KeyError:
                examples[key] = [example[key]]
    np_dataset = {}
    for key in examples.keys():
        np_dataset[key] = np.stack(examples[key])
        np_dataset[key] = np_dataset[key]
    return np_dataset


def change_range(dataset, start=(0., 255.), end=(0., 1.)):
    dataset['image'] = (dataset['image'] - start[0]) / (start[1] - start[0])
    dataset['image'] = dataset['image'] * (end[1] - end[0]) + start[0]
    return dataset


def convert_to_one_hot(dataset, num_labels):
    one_hot = np.zeros(shape=[len(dataset['label']), num_labels])
    one_hot[np.arange(len(dataset['label'])), dataset['label'].astype(np.int)] = 1
    dataset['label'] = one_hot
    return dataset


def generate_labelled_and_unlabelled_datasets(args, dataset):
    dataset = unison_shuffle(args, dataset)
    labelled = {}
    unlabelled = {}
    for key in dataset.keys():
        labelled[key] = dataset[key][:args['labelled_examples']]
        unlabelled[key] = dataset[key][args['labelled_examples']:]
    unlabelled['label'] = np.zeros(shape=unlabelled['label'].shape)
    return labelled, unlabelled


def unison_shuffle(args, dataset):
    assert len(set([len(dataset[key]) for key in dataset.keys()])) == 1
    np.random.seed(args['seed'])
    p = np.random.permutation(len(dataset[list(dataset.keys())[0]]))
    for key in dataset.keys():
        dataset[key] = dataset[key][p]
    return dataset
