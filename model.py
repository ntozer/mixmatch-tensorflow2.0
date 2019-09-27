import tensorflow as tf


class Residual3x3Unit(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, stride, droprate=0., activate_before_residual=False):
        super(Residual3x3Unit, self).__init__()
        self.bn_0 = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.relu_0 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv_0 = tf.keras.layers.Conv2D(channels_out, kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.relu_1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv_1 = tf.keras.layers.Conv2D(channels_out, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.downsample = channels_in != channels_out
        self.shortcut = tf.keras.layers.Conv2D(channels_out, kernel_size=1, strides=stride, use_bias=False)
        self.activate_before_residual = activate_before_residual
        self.dropout = tf.keras.layers.Dropout(rate=droprate)
        self.droprate = droprate

    @tf.function
    def call(self, x, training=True):
        if self.downsample and self.activate_before_residual:
            x = self.relu_0(self.bn_0(x, training=training))
        elif not self.downsample:
            out = self.relu_0(self.bn_0(x, training=training))
        out = self.relu_1(self.bn_1(self.conv_0(x if self.downsample else out), training=training))
        if self.droprate > 0.:
            out = self.dropout(out)
        out = self.conv_1(out)
        return out + (self.shortcut(x) if self.downsample else x)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, n_units, channels_in, channels_out, unit, stride, droprate=0., activate_before_residual=False):
        super(ResidualBlock, self).__init__()
        self.units = self._build_unit(n_units, unit, channels_in, channels_out, stride, droprate, activate_before_residual)

    def _build_unit(self, n_units, unit, channels_in, channels_out, stride, droprate, activate_before_residual):
        units = []
        for i in range(n_units):
            units.append(unit(channels_in if i == 0 else channels_out, channels_out, stride if i == 0 else 1, droprate, activate_before_residual))
        return units

    @tf.function
    def call(self, x, training=True):
        for unit in self.units:
            x = unit(x, training=training)
        return x


class WideResNet(tf.keras.Model):
    def __init__(self, num_classes, depth=28, width=2, droprate=0., input_shape=(None, 32, 32, 3), **kwargs):
        super(WideResNet, self).__init__(input_shape, **kwargs)
        assert (depth - 4) % 6 == 0
        N = int((depth - 4) / 6)
        channels = [16, 16 * width, 32 * width, 64 * width]

        self.conv_0 = tf.keras.layers.Conv2D(channels[0], kernel_size=3, strides=1, padding='same', use_bias=False)
        self.block_0 = ResidualBlock(N, channels[0], channels[1], Residual3x3Unit, 1, droprate, True)
        self.block_1 = ResidualBlock(N, channels[1], channels[2], Residual3x3Unit, 2, droprate)
        self.block_2 = ResidualBlock(N, channels[2], channels[3], Residual3x3Unit, 2, droprate)
        self.bn_0 = tf.keras.layers.BatchNormalization(momentum=0.999)
        self.relu_0 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.avg_pool = tf.keras.layers.AveragePooling2D((8, 8), (1, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(num_classes)

    @tf.function
    def call(self, inputs, training=True):
        x = inputs
        x = self.conv_0(x)
        x = self.block_0(x, training=training)
        x = self.block_1(x, training=training)
        x = self.block_2(x, training=training)
        x = self.relu_0(self.bn_0(x, training=training))
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
