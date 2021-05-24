import functools
import tensorflow as tf


class MLP(tf.keras.Model):

    def __init__(self, target_shape, **kwargs):
        super().__init__()

        units = kwargs.get('units', 256)
        dropout = kwargs.get('dropout', 0.2)
        activation = kwargs.get('activation', 'relu')
        output_raw = kwargs.get('output_raw', False)

        num_classes = target_shape[-1]
        dense = functools.partial(tf.keras.layers.Dense, activation=activation)
        dropout = functools.partial(tf.keras.layers.Dropout, rate=float(dropout))

        self.dense_layers = []
        if isinstance(units, int):
            self.dense_layers.append(dense(units))
            self.dense_layers.append(dropout())
        else:
            for unit in units:
                self.dense_layers.append(dense(unit))
                self.dense_layers.append(dropout())

        if output_raw:
            self.output_layer = tf.keras.layers.Dense(num_classes, activation=None)
        else:
            if num_classes > 1:
                self.output_layer = tf.keras.layers.Dense(num_classes, activation='sigmoid')
            else:
                self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.dense_layers[0](inputs)
        for i in range(1, len(self.dense_layers)):
            x = self.dense_layers[i](x)
        x = self.output_layer(x)
        return x
