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