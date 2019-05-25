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

    train = convert_to_one_hot(train, num_labels)
    test = convert_to_one_hot(test, num_labels)
    trainX, trainU = generate_labelled_and_unlabelled_datasets(args, train)

    return trainX, trainU, test


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
        # if key == 'label':
        #     np_dataset[key] = np.expand_dims(np_dataset[key], axis=1)
    return np_dataset


def convert_to_one_hot(dataset, num_labels):
    one_hot = np.zeros(shape=[len(dataset['label']), num_labels])
    one_hot[np.arange(len(dataset['label'])), dataset['label']] = 1
    dataset['label'] = one_hot
    return dataset


def generate_labelled_and_unlabelled_datasets(args, dataset):
    dataset = unison_shuffle(args, dataset)
    labelled = {}
    unlabelled = {}
    for key in dataset.keys():
        labelled[key] = dataset[key][:args['labelled_examples']]
        unlabelled[key] = dataset[key][args['labelled_examples']:]
    for key in dataset.keys():
        labelled[key] = np.resize(dataset[key], unlabelled[key].shape)
    unlabelled['label'] = np.zeros(shape=unlabelled['label'].shape)
    return labelled, unlabelled


def unison_shuffle(args, dataset):
    assert len(set([len(dataset[key]) for key in dataset.keys()])) == 1
    np.random.seed(args['seed'])
    p = np.random.permutation(len(dataset[list(dataset.keys())[0]]))
    for key in dataset.keys():
        dataset[key] = dataset[key][p]
    return dataset
