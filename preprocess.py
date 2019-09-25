import numpy as np
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
    for key in test.keys():
        test[key] = test[key].astype(dtype=np.float32)
    train, validation = split_dataset(int(0.9 * len(train['label'])), train, num_labels)
    trainX, trainU = split_dataset(args['labelled_examples'], train, num_labels)
    labelled_dist = []
    # outputting the labelled data distribution
    for i in range(num_labels):
        labelled_dist.append(len(np.where(trainX['label'][:, i] == 1)[0]))
    print('Labelled Distribution:', labelled_dist)

    for key in train.keys():
        trainX[key] = trainX[key].astype(dtype=np.float32)
        trainU[key] = trainU[key].astype(dtype=np.float32)
        validation[key] = validation[key].astype(dtype=np.float32)
    test = change_range(test)
    validation = change_range(validation)
    trainX = change_range(trainX)
    trainU = change_range(trainU)
    return trainX, trainU, validation, test, num_labels


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


def change_range(dataset, start=(0., 255.), end=(-1., 1.)):
    dataset['image'] = (dataset['image'] - start[0]) / (start[1] - start[0])
    dataset['image'] = dataset['image'] * (end[1] - end[0]) + start[0]
    return dataset


def convert_to_one_hot(dataset, num_labels):
    one_hot = np.zeros(shape=[len(dataset['label']), num_labels])
    one_hot[np.arange(len(dataset['label'])), dataset['label'].astype(np.int)] = 1
    dataset['label'] = one_hot
    return dataset


def split_dataset(split_size, dataset, num_labels):
    dataset = unison_shuffle(dataset)
    train_idx = []
    validation_idxs = []
    for i in range(num_labels):
        idxs = np.where(dataset['label'][:, i] == 1)[0]
        np.random.shuffle(idxs)
        split_idx = int(split_size / num_labels)
        train_idx.extend(idxs[:split_idx])
        validation_idxs.extend(idxs[split_idx:])
    np.random.shuffle(train_idx)
    np.random.shuffle(validation_idxs)
    train = {}
    validation = {}
    for key in dataset.keys():
        train[key] = dataset[key][train_idx]
        validation[key] = dataset[key][validation_idxs]
    return train, validation


def unison_shuffle(dataset):
    assert len(set([len(dataset[key]) for key in dataset.keys()])) == 1
    p = np.random.permutation(len(dataset[list(dataset.keys())[0]]))
    for key in dataset.keys():
        dataset[key] = dataset[key][p]
    return dataset
