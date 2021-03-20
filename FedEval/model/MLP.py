import functools
import tensorflow as tf


class MLP(tf.keras.Model):

    def __init__(self, target_shape, units, dropout, activation, **kwargs):
        super().__init__()

        num_classes = target_shape[-1]
        dense = functools.partial(tf.keras.layers.Dense, activation=activation)
        self.dropout = functools.partial(tf.keras.layers.Dropout, rate=float(dropout))

        self.dense_layers = []
        if isinstance(units, int):
            self.dense_layers.append(dense(units))
        else:
            for unit in units:
                self.dense_layers.append(dense(unit))
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.dense_layers[0](inputs)
        for i in range(1, len(self.dense_layers)):
            x = self.dense_layers[i](x)
            # x = self.dropout()(x)
        x = self.output_layer(x)
        return x
