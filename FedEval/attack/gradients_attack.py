import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class AttackModel:

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
                self.model.output[target_name] = target_tensor[target_name].name

            loss = self.model.forward(inputs=input_tensor, targets=target_tensor, trainable=False)

            assert loss is not None

            gradients = [[e, tf.gradients(loss, e)[0]] for e in tf.global_variables()
                         if str_match_list(e.name, gradient_names)]

            optimizer = tf.train.GradientDescentOptimizer(lr or self.model.lr)

            attack_loss = []
            for v, g in gradients:
                tmp_g = tf.compat.v1.placeholder(tf.float32, g.shape, g.name.split(':')[0])
                self.model.input[v.name] = tmp_g.name
                self.model.output[v.name] = g.name
                self.model.output[v.name + '_y'] = tf.gradients(tf.reduce_sum(tf.square(g)), target_tensor['y'])[0].name
                assert g.shape == tmp_g.shape
                attack_loss.append(tf.reduce_sum(tf.square(g - tmp_g)))

            attack_loss = tf.reduce_sum(attack_loss)

            mask_tensor_x = tf.concat([tf.zeros([1] + list(self.model.inputs_shape['x'])),
                                      tf.ones([1] + list(self.model.inputs_shape['x']))], axis=0)

            mask_tensor_y = tf.concat([tf.zeros([1] + list(self.model.targets_shape['y'])),
                                       tf.ones([1] + list(self.model.targets_shape['y']))], axis=0)

            batch_index = tf.compat.v1.placeholder(tf.int32, [self.batch_size, ], name='batch_index')
            self.model.input['batch_index'] = batch_index.name
            update_mask_x = tf.gather(mask_tensor_x, batch_index, axis=0)
            update_mask_y = tf.gather(mask_tensor_y, batch_index, axis=0)

            self.model.output['attack_loss'] = attack_loss.name
            self.model.output['loss'] = loss.name

            optimizer_gradients = optimizer.compute_gradients(attack_loss, [input_tensor['x'], target_tensor_raw['y']])
            optimizer_gradients = [list(e) for e in optimizer_gradients]

            optimizer_gradients[0][0] = tf.multiply(update_mask_x, optimizer_gradients[0][0])
            optimizer_gradients[1][0] = tf.multiply(update_mask_y, optimizer_gradients[1][0])

            attack_train_op = optimizer.apply_gradients(optimizer_gradients)
            self.model.op['attack_train_op'] = attack_train_op.name

            ########################
            # TMP
            self.model.output['grad_x'] = tf.gradients(attack_loss, input_tensor['x'])[0].name
            self.model.output['grad_y'] = tf.gradients(attack_loss, target_tensor['y'])[0].name
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