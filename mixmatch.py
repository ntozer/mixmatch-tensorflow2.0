import numpy as np
import tensorflow as tf


def augment_fn(x):
    x = tf.image.random_flip_left_right(x)
    return x


def guess_labels(u_aug, model, K):
    u_logits = model(u_aug[0])
    for k in range(1, K):
        u_logits = u_logits + model(u_aug[k])
    u_logits = u_logits / K
    return u_logits


def sharpen(p, T):
    return tf.pow(p, 1/T) / tf.reduce_sum(tf.pow(p, 1/T))


def mixup(x1, x2, y1, y2, alpha):
    beta = np.random.beta(alpha, alpha)
    beta = tf.maximum(beta, 1-beta)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


def mixmatch(x, u, y, model, T=0.5, K=2, alpha=0.75):
    x = tf.dtypes.cast(x, tf.float32)
    u = tf.dtypes.cast(u, tf.float32)
    y = tf.dtypes.cast(y, tf.float32)
    x_aug = augment_fn(x)
    u_aug = [None for _ in range(K)]
    for k in range(K):
        u_aug[k] = augment_fn(u)
    mean_logits = guess_labels(u_aug, model, K)
    qb = sharpen(mean_logits, T)
    U = tf.concat(u_aug, axis=0)
    qb = tf.concat([qb for _ in range(K)], axis=0)
    W = tf.concat([x_aug, U], axis=0)
    Wy = tf.concat([y, qb], axis=0)
    indices = tf.random.shuffle(tf.range(W.shape[0]))
    W = tf.gather(W, indices)
    Wy = tf.gather(Wy, indices)
    Xp = [mixup(x, w, y, wy, alpha) for x, w, y, wy in zip(x_aug, W[:x_aug.shape[0]], y, Wy[:x_aug.shape[0]])]
    Up = [mixup(x, w, uy, wy, alpha) for x, w, uy, wy in zip(U, W[-1 * U.shape[0]:], qb, Wy[-1 * U.shape[0]:])]
    X = tf.stack([x[0] for x in Xp], axis=0)
    Xy = tf.stack([x[1] for x in Xp], axis=0)
    U = tf.stack([u[0] for u in Up], axis=0)
    Uy = tf.stack([u[1] for u in Up], axis=0)
    return X, Xy, U, Uy
