import tensorflow as tf
from .BaseModel import TFModel


class MLP(TFModel):

    def __init__(self,
                 inputs_shape,
                 targets_shape,
                 units,
                 dropout=0.1,
                 lr=1e-4,
                 optimizer='adam',
                 activation='relu',
                 code_version='MLP',
                 model_dir='log',
                 gpu_device='0'):

        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.lr = float(lr)
        self.optimizer_name = optimizer

        super().__init__(inputs_shape, targets_shape, code_version, model_dir, gpu_device)

    def forward(self, inputs, targets, trainable):

        with self.graph.as_default():

            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(self.units[0], activation=self.activation,
                                            input_shape=(self.inputs_shape['x'][-1],), trainable=trainable))
            model.add(tf.keras.layers.Dropout(0.2))
            for i in range(1, len(self.units)):
                model.add(tf.keras.layers.Dense(self.units[i], activation=self.activation, trainable=trainable))
                model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(self.targets_shape['y'][-1], trainable=trainable))

            y_hat = model.call(inputs['x'])

            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets['y'], logits=y_hat)
            acc = tf.keras.metrics.categorical_accuracy(y_true=targets['y'], y_pred=y_hat)

            if self.optimizer_name.lower() == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.lr)
            elif self.optimizer_name.lower() == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
            elif self.optimizer_name.lower() == 'sgd':
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise ValueError('Do not support the optimizer: ' + self.optimizer_name)

            train_op = self.optimizer.minimize(loss)

            self.output['prediction'] = y_hat.name
            self.output['loss'] = loss.name
            self.output['accuracy'] = acc.name

            self.op['train_op'] = train_op.name

        return loss


class MLPAttack(TFModel):

    def __init__(self,
                 inputs_shape,
                 targets_shape,
                 units,
                 dropout=0.1,
                 lr=1e-4,
                 optimizer='adam',
                 activation='relu',
                 code_version='MLP',
                 model_dir='log',
                 gpu_device='0'):

        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.optimizer_name = optimizer

        super().__init__(inputs_shape, targets_shape, code_version, model_dir, gpu_device)

    def forward(self, inputs, targets, trainable):

        with self.graph.as_default():
            middle_result = inputs['x']
            for layer_index in range(len(self.units)):
                middle_result = self.dense(middle_result,
                                           units=self.units[layer_index], activation=self.activation,
                                           name='dense%s' % layer_index, trainable=trainable)
                # middle_result = tf.nn.dropout(middle_result, keep_prob=1 - self.dropout)

            y_hat = self.dense(middle_result, units=self.targets_shape['y'][-1], activation=None,
                               name='dense%s' % len(self.units), trainable=trainable)

            loss = self.softmax_cross_entropy(labels=targets['y'], logits=y_hat)
            accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(targets['y'], axis=1), tf.argmax(y_hat, axis=1)), tf.float32))

            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            train_op = self.optimizer.minimize(loss)

            self.output['loss'] = tf.reduce_mean(loss).name
            self.output['prediction'] = y_hat.name
            self.output['accuracy'] = accuracy.name
            self.op['train_op'] = train_op.name

            return loss
