from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf


@tf.function
def augment(x):
    # random left right flipping
    x = tf.image.random_flip_left_right(x)
    # random pad and crop
    x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode='REFLECT')
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    return x


def guess_labels(u_aug, model, K):
    u_logits = tf.nn.softmax(model(u_aug[0]), axis=1)
    for k in range(1, K):
        u_logits = u_logits + tf.nn.softmax(model(u_aug[k]), axis=1)
    u_logits = u_logits / K
    u_logits = tf.stop_gradient(u_logits)
    return u_logits


@tf.function
def sharpen(p, T):
    return tf.pow(p, 1/T) / tf.reduce_sum(tf.pow(p, 1/T), axis=1, keepdims=True)


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    beta = tf.maximum(beta, 1-beta)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]


def mixmatch(model, x, y, u, T=0.5, K=2, alpha=0.75):
    batch_size = x.shape[0]
    x_aug = augment(x)
    u_aug = [None for _ in range(K)]
    for k in range(K):
        u_aug[k] = augment(u)
    mean_logits = guess_labels(u_aug, model, K)
    qb = sharpen(mean_logits, T)
    U = tf.concat(u_aug, axis=0)
    qb = tf.concat([qb for _ in range(K)], axis=0)
    XU = tf.concat([x_aug, U], axis=0)
    XUy = tf.concat([y, qb], axis=0)
    indices = tf.random.shuffle(tf.range(XU.shape[0]))
    W = tf.gather(XU, indices)
    Wy = tf.gather(XUy, indices)
    XU, XUy = mixup(XU, W, XUy, Wy, alpha)
    XU = tf.split(XU, K + 1, axis=0)
    XU = interleave(XU, batch_size)
    return XU, XUy


@tf.function
def semi_loss(labels_x, logits_x, labels_u, logits_u):
    loss_xe = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_l2u = tf.square(labels_u - tf.nn.softmax(logits_u))
    loss_l2u = tf.reduce_mean(loss_l2u)
    return loss_xe, loss_l2u


def linear_rampup(epoch, rampup_length=16):
    if rampup_length == 0:
        return 1.
    else:
        rampup = np.clip(epoch / rampup_length, 0., 1.)
        return float(rampup)


def weight_decay(model, decay_rate):
    for var in model.trainable_variables:
        var.assign(var * (1 - decay_rate))


class EMA:
    def __init__(self, model, decay_rate=0.999):
        self.shadow = {}
        self.decay_rate = decay_rate
        self.model = model
        self.variable_refs = {var.name: var for var in self.model.trainable_variables}
        self.weights = None
        self.register(model.trainable_variables)

    def register(self, variables):
        for var in variables:
            self.shadow[var.name] = tf.identity(var)

    def apply(self):
        for var in self.model.trainable_variables:
            average = (1 - self.decay_rate) * var + self.decay_rate * self.shadow[var.name]
            self.shadow[var.name] = average

    def __enter__(self):
        # swap model weights to EMA model for validation
        self.weights = {var.name: tf.identity(var) for var in self.model.trainable_variables}
        for name, var in self.shadow.items():
            self.variable_refs[name].assign(var)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # swap model weights back to original model weights
        for name, var in self.weights.items():
            self.variable_refs[name].assign(var)