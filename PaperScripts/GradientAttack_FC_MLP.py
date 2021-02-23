import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from FedEval.model import MLPAttack
from FedEval.dataset import FedImage

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('--num_layers', type=int, default=2)
arg_parse.add_argument('--num_units', type=int, default=512)
arg_parse.add_argument('--training_samples', type=int, default=1)
arg_parse.add_argument('--client_batch_size', type=int, default=1)
arg_parse.add_argument('--fake_samples', type=int, default=1)
arg_parse.add_argument('--attack_epoch', type=int, default=20000)
arg_parse.add_argument('--attack_local_epoch', type=int, default=1)
arg_parse.add_argument('--client_local_epoch', type=int, default=1)
arg_parse.add_argument('--dataset', type=str, default='mnist')
arg_parse.add_argument('--fname', type=str, default='Debug-FC-MLP')

args = arg_parse.parse_args()

image_dir = os.path.join("image", args.fname)

os.makedirs(image_dir, exist_ok=True)

os.system('DEL image\\%s\\attack*.png' % image_dir)
os.system('DEL image\\%s\\real*.png' % image_dir)

batch_size = args.client_batch_size
training_samples = args.training_samples

client_lr = 5e-4
client_epochs = 1
client_local_epoch = args.client_local_epoch

server_lr = 0.1
server_epochs = args.attack_epoch
server_local_epoch = args.attack_local_epoch
server_plot_epochs = 100

num_layers = args.num_layers
num_units = args.num_units

data_generator = FedImage(dataset='mnist', data_dir=None, flatten=True, normalize=True, num_clients=1)

data = data_generator.iid_data(sample_size=300, save_file=False)

# We choose part of the x_train, y_train
x_train = data[0]['x_train'][:args.training_samples]
y_train = data[0]['y_train'][:args.training_samples]

client_model = MLPAttack(
    inputs_shape={'x': x_train.shape[1:]},
    targets_shape={'y': y_train.shape[1:]},
    units=[512, 512],
    activation=tf.nn.relu, dropout=0, lr=client_lr, gpu_device='0'
)

client_model.build(init_vars=True)

for epoch in range(client_epochs):

    weights_1 = client_model.get_weights()

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

    weights_2 = client_model.get_weights()

    gradients = {key: (weights_1[key] - weights_2[key]) / client_lr / client_local_epoch for key in weights_1}

    attack_gradients = gradients['dense0/kernel:0']

    for i in range(len(x_train)):
        plt.imsave(os.path.join(image_dir, 'real-%s.png' % (i)),
                   x_train[i].reshape([28, 28]),
                   cmap='gray')

    error_all = []
    img_list = []
    for i in range(attack_gradients.shape[1]):
        attack = attack_gradients[:, i]
        attack += abs(attack)
        attack -= attack.min()
        if attack.max() > 0:
            attack /= attack.max()
            img_list.append(attack)
            errors = []
            for j in range(len(x_train)):
                errors.append(np.square(attack - x_train[j]).mean())
            fake_label = np.argmax(y_train[np.argmin(errors)])
            error_all.append(errors)

    error_all = np.array(error_all)
    result_img_list = []
    for i in range(len(x_train)):
        attack_index = np.argmin(error_all[:, i])

        print(i, np.min(error_all[:, i]))

        plt.imsave(os.path.join(image_dir, 'attack-%s.png' % i),
                   img_list[attack_index].reshape([28, 28]),
                   cmap='gray')
        result_img_list.append(img_list[attack_index].reshape([28, 28]))

    plot_image = [x_train, result_img_list]

    Nr, Nc = 2, len(x_train)
    fig, axs = plt.subplots(Nr, Nc, figsize=(Nc, Nr))

    if Nc == 1:
        for i in range(Nr):
            try:
                axs[i].axis('off')
                axs[i].imshow(plot_image[i][0].reshape([28, 28]), cmap='gray')
            except Exception as e:
                print(i, 0, e)
    else:
        for i in range(Nr):
            for j in range(Nc):
                try:
                    axs[i, j].axis('off')
                    axs[i, j].imshow(plot_image[i][j].reshape([28, 28]), cmap='gray')
                except Exception as e:
                    print(i, j, e)
    fig.tight_layout()
    plt.savefig(os.path.join(image_dir, 'attack-result.png'), format="png", dpi=300)
