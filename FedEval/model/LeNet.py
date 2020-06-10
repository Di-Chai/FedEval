import numpy as np
import tensorflow as tf
from .BaseModel import TFModel


class LeNet(TFModel):
    def __init__(self,
                 inputs_shape,
                 targets_shape,
                 lr=1e-4,
                 activation='relu',
                 optimizer='adam',
                 pooling='max',
                 code_version='LeNet',
                 model_dir='log',
                 gpu_device='0'):
        self.activation = activation
        self.optimizer_name = optimizer
        self.pooling = pooling
        self.lr = lr
        super(LeNet, self).__init__(inputs_shape, targets_shape, code_version, model_dir, gpu_device)

    def forward(self, inputs, targets, trainable):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation=self.activation,
                                         input_shape=self.inputs_shape['x'], trainable=trainable))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation=self.activation, trainable=trainable))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(120, activation=self.activation, trainable=trainable))
        model.add(tf.keras.layers.Dense(84, activation=self.activation, trainable=trainable))
        model.add(tf.keras.layers.Dense(self.targets_shape['y'][-1]))
        prediction = model.call(inputs['x'])

        # self defined loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=targets['y'], logits=prediction)
        acc = tf.keras.metrics.categorical_accuracy(y_true=targets['y'], y_pred=prediction)

        self.output['prediction'] = prediction.name
        self.output['loss'] = loss.name
        self.output['accuracy'] = acc.name

        if self.optimizer_name.lower() == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif self.optimizer_name.lower() == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.lr, momentum=0.9)
        elif self.optimizer_name.lower() == 'gd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        else:
            raise ValueError('Do not support the optimizer: ' + self.optimizer_name)

        train_op = self.optimizer.minimize(loss)

        self.op['train_op'] = train_op.name

        return loss


class LeNetAttack(TFModel):
    def __init__(self,
                 inputs_shape,
                 targets_shape,
                 lr=1e-4,
                 activation=tf.nn.relu,
                 code_version='MLP',
                 model_dir='model_dir',
                 gpu_device='-1'):

        self.activation = activation
        self.lr = lr

        super(LeNetAttack, self).__init__(inputs_shape, targets_shape, code_version, model_dir, gpu_device)

    def forward(self, inputs, targets, trainable):

        X = self.conv2d(
            x=inputs['x'],
            filter=(5, 5, inputs['x'].shape[-1].value, 12),
            activation=self.activation,
            padding='SAME',
            stride=(2, 2),
            name='conv1',
            trainable=trainable)

        X = self.conv2d(
            x=X,
            filter=(5, 5, 12, 12),
            activation=self.activation,
            padding='SAME',
            stride=(2, 2),
            name='conv2',
            trainable=trainable)

        X = self.conv2d(
            x=X,
            filter=(5, 5, 12, 12),
            activation=self.activation,
            padding='SAME',
            stride=(1, 1),
            name='conv3',
            trainable=trainable)

        middle_dense = tf.reshape(X, (-1, np.prod(X.get_shape().as_list()[1:])), name='Feature')
        prediction = self.dense(middle_dense, self.targets_shape['y'][-1], activation=None,
                                name='prediction', trainable=trainable)

        loss = self.softmax_cross_entropy(labels=targets['y'], logits=prediction)

        correct_prediction = tf.equal(tf.argmax(targets['y'], 1), tf.argmax(prediction, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction, name='accuracy')

        for v in tf.trainable_variables():
            self.output[v.name] = tf.gradients(loss, v)[0].name

        self.output['prediction'] = prediction.name
        self.output['loss'] = tf.reduce_mean(loss).name
        self.output['accuracy'] = accuracy.name

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)

        train_op_adam = self.optimizer.minimize(loss)

        self.op['train_op'] = train_op_adam.name

        return loss