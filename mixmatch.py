import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def augment_fn(x):
    # random left right flipping
    x = tf.image.random_flip_left_right(x)
    # random pad and crop
    x = tf.pad(x, paddings=[(4, 4), (4, 4), (0, 0)])
    x = tf.image.random_crop(x, size=(32, 32, 3))
    return x


def guess_labels(u_aug, model, K):
    u_logits = tf.nn.softmax(model(u_aug[0]))
    for k in range(1, K):
        u_logits = u_logits + tf.nn.softmax(model(u_aug[k]))
    u_logits = u_logits / K
    u_logits = tf.stop_gradient(u_logits)
    return u_logits


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
    x = tf.dtypes.cast(x, tf.float32)
    u = tf.dtypes.cast(u, tf.float32)
    y = tf.dtypes.cast(y, tf.float32)
    batch_size = x.shape[0]
    x_aug = tf.map_fn(augment_fn, x)
    u_aug = [None for _ in range(K)]
    for k in range(K):
        u_aug[k] = tf.map_fn(augment_fn, u)
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


def semi_loss(labels_x, logits_x, labels_u, logits_u, lambda_u):
    xe_loss = K.mean(-1 * K.sum(labels_x * tf.nn.log_softmax(logits_x), axis=1))
    mse_loss = K.mean(K.square(labels_u - K.softmax(logits_u)))
    semi_loss = xe_loss + lambda_u * mse_loss
    return semi_loss


def linear_rampup(epoch, rampup_length=16):
    if rampup_length == 0:
        return 1.
    else:
        rampup = np.clip(epoch / rampup_length, 0., 1.)
        return float(rampup)


def grad(model, X, y, U, lambda_u, batch_size):
    with tf.GradientTape() as tape:
        XU, XUy = mixmatch(model, X, y, U, )
        logits = [model(XU[0])]
        for batch in XU[1:]:
            logits.append(model(batch))
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = tf.concat(logits[1:], axis=0)
        loss_value = semi_loss(XUy[:batch_size], logits_x, XUy[batch_size:], logits_u, lambda_u)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)
