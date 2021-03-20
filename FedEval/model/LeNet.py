import tensorflow as tf


class LeNet(tf.keras.Model):

    def __init__(self, target_shape, activation, **kwargs):
        super().__init__()

        num_classes = target_shape[-1]
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation=activation)
        self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation=activation)
        self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation=activation)
        self.dense2 = tf.keras.layers.Dense(84, activation=activation)
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
