import argparse
import os

import tensorflow as tf
import yaml

from mixmatch import mixmatch, semi_loss
from model import WideResNet
from preprocess import load_data


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


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
    parser.add_argument('--decay-rate', type=float, default=0., help='decay rate for learning rate (default: 0.)')

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

    trainX, trainU, test, num_classes = load_data(args)

    datasetX = tf.data.Dataset.from_tensor_slices(trainX)
    datasetU = tf.data.Dataset.from_tensor_slices(trainU)
    batchedX = datasetX.batch(args['batch_size'] // 2, drop_remainder=True)
    model = WideResNet(num_classes, depth=28, width=2)

    def grad(model, X, y, U, q, mse_weight):
        with tf.GradientTape() as tape:
            loss_value = semi_loss(X, y, U, q, model, mse_weight)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(lr=args['learning_rate'])
    for epoch in range(args['epochs']):
        optimizer.lr = optimizer.lr * (1 - args['decay_rate'])**(epoch / args['epochs'])

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        batchedU = datasetU.shuffle(buffer_size=int(1e6)).batch(args['batch_size'] // 2, drop_remainder=True)

        for batchX, batchU in zip(batchedX, batchedU):
            X, y, U, q = mixmatch(batchX['image'], batchU['image'], batchX['label'], model)

            loss_value, grads = grad(model, X, y, U, q, args['lambdaU'])
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)
            epoch_accuracy(tf.argmax(batchX['label'], axis=1, output_type=tf.int32), model(tf.cast(batchX['image'], dtype=tf.float32)))

        if epoch % 10 == 0:
            print(f'Epoch {epoch:04d}: Loss: {epoch_loss_avg.result():.3f}, Accuracy: {epoch_accuracy.result():.3%}')

            test_accuracy = tf.keras.metrics.Accuracy()
            test_dataset = tf.data.Dataset.from_tensor_slices(test)
            test_dataset = test_dataset.batch(args['batch_size'])
            for batch in test_dataset:
                logits = model(batch['image'])
                prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
                test_accuracy(prediction, tf.argmax(batch['label'], axis=1, output_type=tf.int32))
            print(f'Test set accuracy: {test_accuracy.result():.3%}')


if __name__ == '__main__':
    main()