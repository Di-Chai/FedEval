import tensorflow as tf


class LeNet(tf.keras.Model):

    def __init__(self, target_shape, activation, **kwargs):
        super().__init__()

        output_raw = kwargs.get('output_raw', False)
        self.pooling = kwargs.get('use_pooling', 'max')

        num_classes = target_shape[-1]
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation=activation)
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation=activation)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120, activation=activation)
        self.dense2 = tf.keras.layers.Dense(84, activation=activation)

        if self.pooling is not None:
            if self.pooling == 'max':
                self.pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
                self.pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
            else:
                self.pooling1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
                self.pooling2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))

        if output_raw:
            self.dense3 = tf.keras.layers.Dense(num_classes, activation=None)
        else:
            if num_classes > 1:
                self.dense3 = tf.keras.layers.Dense(num_classes, activation='sigmoid')
            else:
                self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        if self.pooling is not None:
            x = self.pooling1(x)
        x = self.conv2(x)
        if self.pooling is not None:
            x = self.pooling2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
