import tensorflow as tf


class StackedLSTM(tf.keras.Model):

    def __init__(self, target_shape, embedding_dim, hidden_units=256, **kwargs):
        super().__init__()

        num_classes = target_shape[-1]
        if embedding_dim > 0:
            self.embedding = tf.keras.layers.Embedding(num_classes, embedding_dim)
        else:
            self.embedding = None
        self.lstm_1 = tf.keras.layers.LSTM(hidden_units, return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(hidden_units, return_sequences=False)
        if num_classes > 1:
            self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')
        else:
            self.dense = tf.keras.layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        if self.embedding is not None:
            x = self.embedding(inputs)
        else:
            x = inputs
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.dense(x)
        return x
