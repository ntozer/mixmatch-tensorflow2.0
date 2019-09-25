import argparse
import os
import time

import tensorflow as tf
import tqdm
import yaml

from mixmatch import mixmatch, semi_loss, linear_rampup, interleave, weight_decay, EMA
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
    parser.add_argument('--tensorboard', action='store_true', help='enable tensorboard visualization')

    return parser.parse_args()


def load_config(args):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(dir_path, args['config_path'])
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    for key in args.keys():
        if key in config.keys():
            args[key] = config[key]
    return args


def main():
    args = vars(get_args())
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if args['config_path'] is not None and os.path.exists(os.path.join(dir_path, args['config_path'])):
        args = load_config(args)

    trainX, trainU, validation, test, num_classes = load_data(args)

    datasetX = tf.data.Dataset.from_tensor_slices(trainX)
    datasetU = tf.data.Dataset.from_tensor_slices(trainU)

    val_dataset = tf.data.Dataset.from_tensor_slices(validation)
    test_dataset = tf.data.Dataset.from_tensor_slices(test)

    model = WideResNet(num_classes, depth=28, width=2)
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()

    ema = EMA(decay_rate=args['ema_decay'])
    ema.register(model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(lr=args['learning_rate'])

    train_writer = None
    if args['tensorboard']:
        train_writer = tf.summary.create_file_writer(f'.logs/{args["dataset"]}@{args["labelled_examples"]}/{int(time.time())}/train')
        val_writer = tf.summary.create_file_writer(f'.logs/{args["dataset"]}@{args["labelled_examples"]}/{int(time.time())}/validation')
        test_writer = tf.summary.create_file_writer(f'.logs/{args["dataset"]}@{args["labelled_examples"]}/{int(time.time())}/test')

    for epoch in range(args['epochs']):
        xe_loss, l2u_loss, total_loss, accuracy = train(datasetX, datasetU, model, ema, optimizer, epoch, args, train_writer)

        # swap model weights to EMA model for validation
        variable_refs = {var.name: var for var in model.trainable_variables}
        trainable_variables = {var.name: tf.identity(var) for var in model.trainable_variables}
        for name, var in ema.shadow.items():
            variable_refs[name].assign(var)

        val_xe_loss, val_accuracy = validate(val_dataset, model, epoch, args, split='Validation')
        test_xe_loss, test_accuracy = validate(test_dataset, model, epoch, args, split='Test')

        # TODO add model saving code here

        # swap model weights back to original model weights
        for name, var in trainable_variables.items():
            variable_refs[name].assign(var)

        step = args['val_iteration'] * (epoch + 1)

        if args['tensorboard'] is not None:
            with train_writer.as_default():
                tf.summary.scalar('xe_loss', xe_loss.result(), step=step)
                tf.summary.scalar('l2u_loss', l2u_loss.result(), step=step)
                tf.summary.scalar('total_loss', total_loss.result(), step=step)
                tf.summary.scalar('accuracy', accuracy.result(), step=step)
            with val_writer.as_default():
                tf.summary.scalar('xe_loss', val_xe_loss.result(), step=step)
                tf.summary.scalar('accuracy', val_accuracy.result(), step=step)
            with test_writer.as_default():
                tf.summary.scalar('xe_loss', test_xe_loss.result(), step=step)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=step)

    if args['tensorboard']:
        for writer in [train_writer, val_writer, test_writer]:
            writer.flush()


# @tf.function
def train(datasetX, datasetU, model, ema, optimizer, epoch, args, writer):
    xe_loss_avg = tf.keras.metrics.Mean()
    l2u_loss_avg = tf.keras.metrics.Mean()
    total_loss_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    shuffle_and_batch = lambda dataset: dataset.shuffle(buffer_size=int(1e6)).batch(batch_size=args['batch_size'], drop_remainder=True)

    iteratorX = iter(shuffle_and_batch(datasetX))
    iteratorU = iter(shuffle_and_batch(datasetU))

    for batch_num in tqdm.tqdm(range(args['val_iteration']), unit='batch'):
        lambda_u = args['lambda_u'] * linear_rampup(epoch + batch_num/args['val_iteration'], args['rampup_length'])
        try:
            batchX = next(iteratorX)
        except:
            iteratorX = iter(shuffle_and_batch(datasetX))
            batchX = next(iteratorX)
        try:
            batchU = next(iteratorU)
        except:
            iteratorU = iter(shuffle_and_batch(datasetU))
            batchU = next(iteratorU)

        with tf.GradientTape() as tape:
            # run mixmatch
            XU, XUy = mixmatch(model, batchX['image'], batchX['label'], batchU['image'], args['T'], args['K'], args['alpha'])
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
        ema.apply(variables=model.trainable_variables)
        weight_decay(model=model, decay_rate=args['weight_decay'] * args['learning_rate'])

        xe_loss_avg(xe_loss)
        l2u_loss_avg(l2u_loss)
        total_loss_avg(total_loss)
        accuracy(tf.argmax(batchX['label'], axis=1, output_type=tf.int32), model(tf.cast(batchX['image'], dtype=tf.float32), training=False))

    if writer is not None:
        step = args['val_iteration'] * (epoch + 1)
        with writer.as_default():
            tf.summary.image('batchX', batchX['image'], step, 4)
            tf.summary.image('batchU', batchU['image'], step, 4)
            tf.summary.image('mixmatch', XU[0], step, 4)

    print(f'Epoch {epoch:04d}: XE Loss: {xe_loss_avg.result():.4f}, L2U Loss: {l2u_loss_avg.result():.4f}, WeightU: {lambda_u:.2f}, Total Loss: {total_loss_avg.result():.4f}, Accuracy: {accuracy.result():.3%}')
    return xe_loss_avg, l2u_loss_avg, total_loss_avg, accuracy


def validate(dataset, model, epoch, args, split):
    accuracy = tf.keras.metrics.Accuracy()
    xe_avg = tf.keras.metrics.Mean()

    dataset = dataset.batch(args['batch_size'])
    for batch in dataset:
        logits = model(batch['image'], training=False)
        xe_loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch['label'], logits=logits)
        xe_avg(xe_loss)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
        accuracy(prediction, tf.argmax(batch['label'], axis=1, output_type=tf.int32))
    print(f'Epoch {epoch:04d}: {split} XE Loss: {xe_avg.result():.4f}, {split} Accuracy: {accuracy.result():.3%}')
    return xe_avg, accuracy


if __name__ == '__main__':
    main()
