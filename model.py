import tensorflow as tf


class WideResNetBlock(tf.keras.layers.Layer):
    def __init__(self, block_id, filters, kernel=(3, 3), strides=(1, 1), **kwargs):
        super(WideResNetBlock, self).__init__(**kwargs)
        self.filters = filters
        self.strides = [(val, val) for val in strides]
        self.bn_0 = tf.keras.layers.BatchNormalization()
        self.conv2d_0 = tf.keras.layers.Conv2D(filters, kernel, self.strides[0], padding='same')
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.conv2d_1 = tf.keras.layers.Conv2D(filters, kernel, self.strides[1], padding='same')
        self.downsample = None
        if block_id == 0:
            self.downsample = {
                'conv2d': tf.keras.layers.Conv2D(self.filters, (1, 1), self.strides[0], padding='same'),
                'max-pool': tf.keras.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
            }

    def call(self, inputs, **kwargs):
        shortcut = inputs
        x = self.bn_0(tf.cast(inputs, tf.float32))
        x = tf.keras.activations.relu(x)
        if self.downsample is not None:
            if self.filters == inputs.shape[-1]:
                if self.strides[0] == (2, 2):
                    shortcut = self.downsample['max-pool'](x)
            else:
                shortcut = self.downsample['conv2d'](x)
        x = self.conv2d_0(x)
        x = self.bn_1(tf.cast(x, tf.float32))
        x = tf.keras.activations.relu(x)
        x = self.conv2d_1(x)
        return x + shortcut


class WideResNet(tf.keras.Model):
    def __init__(self, num_classes, depth=28, width=2, input_shape=(None, 32, 32, 3), **kwargs):
        super(WideResNet, self).__init__(input_shape, **kwargs)
        assert (depth - 4) % 6 == 0
        N = int((depth - 4) / 6)
        self.groups = [
            [tf.keras.layers.Conv2D(16, (3, 3), (1, 1), padding='same')],
            [WideResNetBlock(num, 16 * width, (3, 3), (1, 1)) for num in range(N)],
            [WideResNetBlock(num, 32 * width, (3, 3), (2, 1) if num == 0 else (1, 1)) for num in range(N)],
            [WideResNetBlock(num, 64 * width, (3, 3), (2, 1) if num == 0 else (1, 1)) for num in range(N)]
        ]
        self.avg_pool = tf.keras.layers.AveragePooling2D((8, 8), (1, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, **kwargs):
        x = inputs
        for group in self.groups:
            for block in group:
                x = block(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
