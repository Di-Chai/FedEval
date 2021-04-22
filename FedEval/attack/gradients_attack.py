import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class AttackModel_Old:

    def __init__(self, model, batch_size, raw_shape):
        self.model = model
        self.batch_size = batch_size
        self.raw_shape = raw_shape

    def build_attack(self, gradient_names, init_vars=True, max_to_keep=5, lr=None):
        def str_match_list(v, vl):
            for e in vl:
                if e in v:
                    return True
            return False

        with self.model.graph.as_default():

            input_tensor = {}
            for input_name in self.model.inputs_shape:
                input_tensor[input_name] = \
                    tf.Variable(
                        tf.random.normal(
                            [self.batch_size] + list(self.model.inputs_shape[input_name]),
                            dtype=tf.float32),
                        name=input_name
                    )
                self.model.output[input_name] = input_tensor[input_name].name

            target_tensor = {}
            target_tensor_raw = {}
            for target_name in self.model.targets_shape:
                target_tensor_raw[target_name] = tf.Variable(tf.random.normal(
                    [self.batch_size] + list(self.model.targets_shape[target_name]),
                    dtype=tf.float32),
                    name=target_name)
                target_tensor[target_name] = tf.nn.softmax(target_tensor_raw[target_name], axis=-1)
                self.model.output[target_name] = target_tensor[target_name].cid

            loss = self.model.forward(inputs=input_tensor, targets=target_tensor, trainable=False)

            assert loss is not None

            gradients = [[e, tf.gradients(loss, e)[0]] for e in tf.global_variables()
                         if str_match_list(e.cid, gradient_names)]

            optimizer = tf.train.GradientDescentOptimizer(lr or self.model.lr)

            attack_loss = []
            for v, g in gradients:
                tmp_g = tf.compat.v1.placeholder(tf.float32, g.shape, g.cid.split(':')[0])
                self.model.input[v.cid] = tmp_g.cid
                self.model.output[v.cid] = g.cid
                self.model.output[v.cid + '_y'] = tf.gradients(tf.reduce_sum(tf.square(g)), target_tensor['y'])[0].cid
                assert g.shape == tmp_g.shape
                attack_loss.append(tf.reduce_sum(tf.square(g - tmp_g)))

            attack_loss = tf.reduce_sum(attack_loss)

            mask_tensor_x = tf.concat([tf.zeros([1] + list(self.model.inputs_shape['x'])),
                                      tf.ones([1] + list(self.model.inputs_shape['x']))], axis=0)

            mask_tensor_y = tf.concat([tf.zeros([1] + list(self.model.targets_shape['y'])),
                                       tf.ones([1] + list(self.model.targets_shape['y']))], axis=0)

            batch_index = tf.compat.v1.placeholder(tf.int32, [self.batch_size, ], name='batch_index')
            self.model.input['batch_index'] = batch_index.cid
            update_mask_x = tf.gather(mask_tensor_x, batch_index, axis=0)
            update_mask_y = tf.gather(mask_tensor_y, batch_index, axis=0)

            self.model.output['attack_loss'] = attack_loss.cid
            self.model.output['loss'] = loss.cid

            optimizer_gradients = optimizer.compute_gradients(attack_loss, [input_tensor['x'], target_tensor_raw['y']])
            optimizer_gradients = [list(e) for e in optimizer_gradients]

            optimizer_gradients[0][0] = tf.multiply(update_mask_x, optimizer_gradients[0][0])
            optimizer_gradients[1][0] = tf.multiply(update_mask_y, optimizer_gradients[1][0])

            attack_train_op = optimizer.apply_gradients(optimizer_gradients)
            self.model.op['attack_train_op'] = attack_train_op.cid

            ########################
            # TMP
            self.model.output['grad_x'] = tf.gradients(attack_loss, input_tensor['x'])[0].cid
            self.model.output['grad_y'] = tf.gradients(attack_loss, target_tensor['y'])[0].cid
            ########################

            self.model.build_essential(init_vars=init_vars, max_to_keep=max_to_keep)

    def attack(self, gradients, attack_epochs, attack_iterations, plot_epoch=200,
               fake_image_name='x', fake_label_name='y', cmap=None, file_name='attack_result'):

        plot_image = [[]]
        fake_data = self.model.predict(output_names=(fake_image_name, fake_label_name), sample_size=0)
        for batch_index in range(self.batch_size):
            plot_image[0].append(fake_data[fake_image_name][batch_index].reshape(self.raw_shape))

        for server_attack_epoch in range(int(attack_epochs / attack_iterations)):

            tmp_plot = []

            total_epoch = (server_attack_epoch + 1) * attack_iterations

            for batch_index in range(self.batch_size):

                gradients['batch_index'] = np.eye(self.batch_size)[batch_index]

                self.model.fit(
                    train_data=gradients,
                    train_sample_size=1,
                    output_names=('attack_loss',),
                    op_names=('attack_train_op',),
                    evaluate_loss_name='attack_loss',
                    batch_size=self.batch_size,
                    max_epoch=attack_iterations,
                    validate_ratio=0,
                    save_model=False,
                    early_stop_length=100000
                )

                fake_data = self.model.predict(output_names=('x', 'y'), sample_size=0)

                tmp_plot.append(fake_data['x'][batch_index].reshape(self.raw_shape))

            if total_epoch % plot_epoch == 0:
                plot_image.append(tmp_plot)

        if cmap == 'gray':
            for i in range(len(plot_image)):
                for j in range(len(plot_image[i])):
                    plot_image[i][j] = np.mean(plot_image[i][j], axis=-1)

        Nr, Nc = max(len(plot_image[0]), self.batch_size), len(plot_image)
        fig, axs = plt.subplots(Nr, Nc, figsize=(Nc, Nr))

        if Nr > 1:
            for i in range(Nr):
                for j in range(Nc):
                    try:
                        axs[i, j].axis('off')
                        axs[i, j].imshow(plot_image[j][i], cmap=cmap)
                    except Exception as e:
                        print(i, j, e)
        else:
            for j in range(Nc):
                axs[j].imshow(plot_image[j][0], cmap=cmap)
                axs[j].axis('off')
        fig.tight_layout()
        plt.savefig(file_name, type="png", dpi=300)


class DLGAttack(tf.keras.Model):
    def __init__(self, base_model, num_samples, fake_data_shape, fake_label_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.num_samples = num_samples
        self.fake_data_shape = fake_data_shape
        self.fake_label_shape = fake_label_shape
        single_image_shape = [1] + list(fake_data_shape)
        single_label_shape = [1] + list(fake_label_shape)
        self.fake_data = [
            tf.Variable(tf.random.normal(single_image_shape), dtype=tf.float32, trainable=True)
            for _ in range(num_samples)
        ]
        self.fake_label = [
            tf.Variable(tf.random.normal(single_label_shape), dtype=tf.float32, trainable=True)
            for _ in range(num_samples)
        ]

    @staticmethod
    def loss_cross_entropy(y_true, y_pred, sigmoid_to_pred=True):
        if sigmoid_to_pred:
            return tf.reduce_mean(tf.reduce_sum(-y_true * tf.nn.log_softmax(y_pred, axis=-1), axis=-1))
        else:
            return tf.reduce_mean(tf.reduce_sum(-y_true*tf.math.log(y_pred), axis=-1))

    def call(self, inputs, training=None, mask=None):
        with tf.GradientTape() as tape:
            y_hat = self.base_model(tf.concat(self.fake_data, axis=0), training=False)
            loss = self.loss_cross_entropy(tf.nn.softmax(tf.concat(self.fake_label, axis=0), axis=-1), y_hat)
            gradients = tape.gradient(loss, self.base_model.variables)

        gradients = [tf.expand_dims(e, axis=0) for e in gradients]
        return tf.reduce_sum([tf.reduce_sum(tf.square(gradients[i] - inputs[i])) for i in range(len(gradients))])

    def attack(self, gradients, init_lr=0.1, epochs=10000, decay_rate=0.98, decay_steps=100,
               data_type='image', output_dir='log/images'):

        optimizer2 = tf.keras.optimizers.Adam(lr=init_lr)
        self.compile(loss='mae', optimizer=optimizer2)

        gradients = [np.expand_dims(e, 0) for e in gradients]

        for epochs in range(epochs):
            loss_list = []
            for image_idx in range(self.num_samples):
                with tf.GradientTape() as tape:
                    loss = self(gradients)
                    target_variables = [
                        self.trainable_variables[i]
                        for i in range(len(self.trainable_variables)) if i % self.num_samples == image_idx
                    ]
                    fake_data_gradients = tape.gradient(loss, target_variables)
                    self.optimizer.apply_gradients(
                        [[fake_data_gradients[i], target_variables[i]] for i in range(len(target_variables))]
                    )
                    loss_list.append(loss)
                self.optimizer.lr = init_lr * np.power(decay_rate, epochs // decay_steps)

            print('Epoch %s loss %s using lr %s' % (epochs, np.mean(loss_list), self.optimizer.lr.numpy()))

            if data_type == 'image' and epochs % 1000 == 0:
                attack_result = np.concatenate(
                    [e.numpy().reshape([1] + self.fake_data_shape)
                     for e in self.trainable_variables[:self.num_samples]], axis=0
                )
                for i in range(attack_result.shape[0]):
                    plt.imshow(attack_result[i])
                    plt.savefig(os.path.join(output_dir, 'fd_%s_%s.png' % (epochs, i)))

        return self.trainable_variables

    def plot_images(self):
        pass
