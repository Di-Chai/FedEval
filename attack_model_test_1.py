import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class MLP(tf.keras.Model):

    def __init__(self, classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = tf.keras.layers.Dense(512)
        self.dense2 = tf.keras.layers.Dense(512)
        self.dense3 = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class AttackModel(tf.keras.Model):
    def __init__(self, base_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.fake_data = tf.Variable(tf.random.normal([2, 28*28], dtype=tf.float32), trainable=True)
        self.fake_label = tf.Variable(tf.random.normal([2, 10], dtype=tf.float32), trainable=True)

    def call(self, inputs, training=None, mask=None):
        with tf.GradientTape() as tape:
            y_hat = self.base_model(self.fake_data, training=False)
            loss = tf.keras.losses.categorical_crossentropy(tf.keras.activations.softmax(self.fake_label), y_hat)
            gradients = tape.gradient(loss, self.base_model.variables)
            gradients = [tf.expand_dims(e, axis=0) for e in gradients]
        return gradients


(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype(np.float32).reshape([-1, 28*28])
y_train = y_train.astype(np.int32)
y_train = np.eye(10)[y_train]

x_train = x_train[:1]
y_train = y_train[:1]

mlp_model = MLP(y_train.shape[-1])
optimizer1 = tf.keras.optimizers.SGD(lr=1e-4)
mlp_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer1)

mlp_model.build((None, x_train.shape[-1]))

with tf.GradientTape() as tape:
    y_hat = mlp_model(x_train)
    loss = tf.keras.losses.categorical_crossentropy(y_train, y_hat)
    gradients = tape.gradient(loss, mlp_model.trainable_variables)


@tf.function
def dgl(x):
    pass

# g_test = [(pre_params[i] - cur_params[i]) / 1e-4 for i in range(len(pre_params))]

# Start attack
attack_model_base = MLP(10)
attack_model_base.build((None, x_train.shape[-1]))
attack_model_base.set_weights(mlp_model.get_weights())
attack_model_base.trainable = False

attack_model = AttackModel(attack_model_base)
optimizer2 = tf.keras.optimizers.SGD(lr=1)
attack_model.compile(loss='mse', optimizer=optimizer2)

attack_model.fit(x=np.zeros([1]), y=[np.expand_dims(e, 0) for e in gradients], epochs=10000, batch_size=1)

plt.imshow(attack_model.trainable_variables[-2][-1].numpy().reshape([28, 28]), cmap='gray')
plt.show()
