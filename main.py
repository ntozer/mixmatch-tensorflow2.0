import argparse
import os
import time

import tensorflow as tf
import tqdm
import yaml

from mixmatch import mixmatch, semi_loss, linear_rampup, interleave, weight_decay, ema_decay
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
    parser.add_argument('--val-iteration', type=int, default=1024, help='number of iterations before validation (default: 1024)')
    parser.add_argument('--T', type=float, default=0.5, help='temperature sharpening ratio (default: 0.5)')
    parser.add_argument('--K', type=int, default=2, help='number of rounds of augmentation (default: 2)')
    parser.add_argument('--alpha', type=float, default=0.75,
                        help='param for sampling from Beta distribution (default: 0.75)')
    parser.add_argument('--lambda-u', type=int, default=100, help='multiplier for unlabelled loss (default: 100)')
    parser.add_argument('--rampup-length', type=int, default=16,
                        help='rampup length for unlabelled loss multiplier (default: 16)')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='decay rate for model vars (default: 0.02)')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='ema decay for ema model vars (default: 0.999)')

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
    if args['config_path'] is not None and os.path.exists(args['config_path']):
        args = load_config(args)

    trainX, trainU, test, num_classes = load_data(args)

    datasetX = tf.data.Dataset.from_tensor_slices(trainX).shuffle(buffer_size=int(1e6), reshuffle_each_iteration=True)
    datasetU = tf.data.Dataset.from_tensor_slices(trainU).shuffle(buffer_size=int(1e6), reshuffle_each_iteration=True)
    datasetX = datasetX.batch(args['batch_size'], drop_remainder=True)
    datasetU = datasetU.batch(args['batch_size'], drop_remainder=True)

    test_dataset = tf.data.Dataset.from_tensor_slices(test)

    model = WideResNet(num_classes, depth=28, width=2)
    ema_model = tf.keras.models.clone_model(model)
    ema_model.set_weights(model.get_weights())

    optimizer = tf.keras.optimizers.Adam(lr=args['learning_rate'])

    writer = tf.summary.create_file_writer(f'.logs/{args["dataset"]}@{args["labelled_examples"]}/{int(time.time())}')

    for epoch in range(args['epochs']):
        xe_loss, l2u_loss, total_loss, accuracy = train(datasetX, datasetU, model, ema_model, optimizer, epoch, args)
        test_xe_loss, test_accuracy = validate(test_dataset, ema_model, epoch, args)

        step = args['val_iteration'] * (epoch + 1)

        with writer.as_default():
            tf.summary.scalar('xe_loss', xe_loss.result(), step=step)
            tf.summary.scalar('l2u_loss', l2u_loss.result(), step=step)
            tf.summary.scalar('total_loss', total_loss.result(), step=step)
            tf.summary.scalar('accuracy', accuracy.result(), step=step)
            tf.summary.scalar('xe_loss(test)', test_xe_loss.result(), step=step)
            tf.summary.scalar('accuracy(test)', test_accuracy.result(), step=step)

    writer.flush()


def train(datasetX, datasetU, model, ema_model, optimizer, epoch, args):
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    iteratorX = iter(datasetX)
    iteratorU = iter(datasetU)

    for batch_num in tqdm.tqdm(range(args['val_iteration'])):
        lambda_u = args['lambda_u'] * linear_rampup(epoch + batch_num/args['val_iteration'], args['rampup_length'])
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(datasetX)
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(datasetU)
            batchU = next(iteratorU)

        with tf.GradientTape() as tape:
            # run mixmatch
            XU, XUy = mixmatch(model, batchX['image'], batchX['label'], batchU['image'])
            logits = [model(XU[0])]
            for batch in XU[1:]:
                logits.append(model(batch))
            logits = interleave(logits, args['batch_size'])
            logits_x = logits[0]
            logits_u = tf.concat(logits[1:], axis=0)

            # compute loss and gradients
            xe_loss, l2u_loss = semi_loss(XUy[:args['batch_size']], logits_x, XUy[args['batch_size']:], logits_u)
            total_loss = xe_loss + lambda_u * l2u_loss
            grads = tape.gradient(total_loss, model.trainable_variables)

        # run optimizer step
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema_decay(model, ema_model, args['ema_decay'])
        weight_decay(model, args['weight_decay'] * args['learning_rate'])

        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        accuracy(tf.argmax(batchX['label'], axis=1, output_type=tf.int32),model(tf.cast(batchX['image'], dtype=tf.float32)))

    print(f'Epoch {epoch:04d}: XE Loss: {xe_loss_avg.result():.3f}, L2U Loss: {l2u_loss_avg.result():.3f}, WeightU: {lambda_u:.2f}, Total Loss: {total_loss_avg.result():.3f}, Accuracy: {accuracy.result():.3%}')
    return xe_loss_avg, l2u_loss_avg, total_loss_avg, accuracy


def validate(dataset, model, epoch, args):
    test_accuracy = tf.keras.metrics.Accuracy()
    test_xe_avg = tf.keras.metrics.Mean()

    dataset = dataset.batch(args['batch_size'])
    for batch in dataset:
        logits = model(batch['image'])
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch['label'], logits=logits)
        test_xe_avg(xe_loss)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        test_accuracy(prediction, tf.argmax(batch['label'], axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: Test XE Loss: {test_xe_avg.result():.3f}, Test Accuracy: {test_accuracy.result():.3%}')
    return test_xe_avg, test_accuracy


if __name__ == '__main__':
    main()
