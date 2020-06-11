import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from FedEval.model import LeNetAttack
from FedEval.dataset import FedImage

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('--training_samples', type=int, default=5)
arg_parse.add_argument('--client_batch_size', type=int, default=5)
arg_parse.add_argument('--fake_samples', type=int, default=5)
arg_parse.add_argument('--attack_epoch', type=int, default=20000)
arg_parse.add_argument('--server_plot_epochs', type=int, default=500)
arg_parse.add_argument('--attack_local_epoch', type=int, default=1)
arg_parse.add_argument('--client_local_epoch', type=int, default=1)
arg_parse.add_argument('--server_lr', type=float, default=0.1)
arg_parse.add_argument('--dataset', type=str, default='mnist')
arg_parse.add_argument('--fname', type=str, default='Debug-DLG-LeNet')

args = arg_parse.parse_args()

if os.path.isdir(os.path.join('image', args.fname)) is False:
    os.makedirs(os.path.join('image', args.fname))

os.system('DEL image\\%s\\fake*.png' % args.fname)
os.system('DEL image\\%s\\real*.png' % args.fname)

batch_size = args.client_batch_size
training_samples = args.training_samples

client_lr = 5e-4
client_epochs = 1
client_local_epoch = args.client_local_epoch

server_lr = args.server_lr
server_epochs = args.attack_epoch
server_local_epoch = args.attack_local_epoch
server_plot_epochs = args.server_plot_epochs

data_generator = FedImage(dataset='mnist', data_dir=None, flatten=False, normalize=True, num_clients=1)

data = data_generator.iid_data(sample_size=300, save_file=False)

# We choose part of the x_train, y_train
x_train = data[0]['x_train'][:args.training_samples]
y_train = data[0]['y_train'][:args.training_samples]

cmap = 'gray'
raw_shape = [28, 28]

client_model = LeNetAttack(
    inputs_shape={'x': x_train.shape[1:]},
    targets_shape={'y': y_train.shape[1:]},
    activation=None,
    lr=client_lr, gpu_device='0'
)

client_model.build(init_vars=True)

server_model = LeNetAttack(
    inputs_shape={'x': x_train.shape[1:]},
    targets_shape={'y': y_train.shape[1:]},
    activation=None,
    lr=server_lr, gpu_device='0'
)

server_model.build_attack(gradient_names=['kernel', 'bias'], batch_size=args.fake_samples, init_vars=True)

for epoch in range(client_epochs):

    weights_1 = client_model.get_weights()['trainable']

    client_model.fit(
        train_data={'x': x_train[:training_samples], 'y': y_train[:training_samples]},
        train_sample_size=training_samples,
        output_names=('loss', 'accuracy'),
        op_names=('train_op',),
        evaluate_loss_name='loss',
        batch_size=batch_size,
        max_epoch=client_local_epoch,
        validate_ratio=0,
        save_model=False,
    )

    weights_2 = client_model.get_weights()['trainable']

    gradients = {key: (weights_1[key] - weights_2[key]) / client_lr / client_local_epoch for key in weights_1}

    server_model.set_weights(weights_1)

    plot_image = [[], []]
    for batch_index in range(training_samples):
        plt.imsave(os.path.join("image", args.fname,
                                'real_image_%s_%s_%s.png' % (0, batch_index, np.argmax(y_train[batch_index]))),
                   x_train[batch_index].reshape(raw_shape), cmap=cmap)
        plot_image[0].append(x_train[batch_index].reshape(raw_shape))

    fake_data = server_model.predict(output_names=('x', 'y'), sample_size=0)
    for batch_index in range(args.fake_samples):
        plt.imsave(
            os.path.join("image", args.fname,
                         "fake_image_%s_%s_%s.png" % (0, batch_index, np.argmax(fake_data['y'][batch_index]))),
            fake_data['x'][batch_index].reshape(raw_shape), cmap=cmap
        )
        plot_image[1].append(fake_data['x'][batch_index].reshape(raw_shape))

    for server_attack_epoch in range(int(server_epochs / server_local_epoch)):

        tmp_plot = []

        total_epoch = (server_attack_epoch + 1) * server_local_epoch

        for batch_index in range(args.fake_samples):

            gradients['batch_index'] = np.eye(args.fake_samples)[batch_index]

            server_model.fit(
                gradients,
                train_sample_size=1,
                output_names=('attack_loss',),
                op_names=('attack_train_op',),
                evaluate_loss_name='attack_loss',
                batch_size=batch_size,
                max_epoch=server_local_epoch,
                validate_ratio=0,
                save_model=False,
                early_stop_length=100000
            )

            fake_data = server_model.predict(output_names=('x', 'y'), sample_size=0)

            tmp_plot.append(fake_data['x'][batch_index].reshape(raw_shape))

            if total_epoch % server_plot_epochs == 0:
                plt.imsave(
                    os.path.join("image", args.fname,
                                 "fake_image_%s_%s_%s.png" %
                                 (total_epoch, batch_index, np.argmax(fake_data['y'][batch_index]))),
                    fake_data['x'][batch_index].reshape(raw_shape), cmap=cmap
                )

        if total_epoch % server_plot_epochs == 0:
            plot_image.append(tmp_plot)

    Nr, Nc = max(len(plot_image[0]), args.fake_samples), len(plot_image)
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
    plt.savefig(os.path.join('image', args.fname, '%s.png' % args.fname), type="png", dpi=300)
