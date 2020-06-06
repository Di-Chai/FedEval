import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from FedEval.model import MLP
from FedEval.dataset import FedImage

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument('--dataset', type=str, default='mnist')
arg_parse.add_argument('--training_samples', type=int, default=10)
arg_parse.add_argument('--client_batch_size', type=int, default=1)
arg_parse.add_argument('--client_local_epoch', type=int, default=1)

args = arg_parse.parse_args()

client_lr = 5e-4

image_dir = os.path.join("image", 'MLP_FC_Attack')
os.makedirs(image_dir, exist_ok=True)
os.system('DEL image\\%s\\*.png' % 'MLP_FC_Attack')

data_generator = FedImage(dataset='mnist', data_dir=None, flatten=True, normalize=True, num_clients=1)

data = data_generator.iid_data(sample_size=300, save_file=False)

# We choose part of the x_train, y_train
x_train = data[0]['x_train'][:args.training_samples]
y_train = data[0]['y_train'][:args.training_samples]

client_model = MLP(inputs_shape={'x': x_train.shape[1:]},
                   targets_shape={'y': y_train.shape[1:]}, optimizer='gd',
                   units=[512, 512], dropout=0, lr=client_lr, gpu_device='0')

client_model.build(init_vars=True)

############################################################################
# Start The attack simulation
############################################################################

weights_1 = client_model.get_weights()['trainable']

client_model.fit(
    train_data={'x': x_train, 'y': y_train},
    train_sample_size=args.training_samples,
    output_names=('loss',),
    op_names=('train_op', ),
    evaluate_loss_name='loss',
    batch_size=args.client_batch_size,
    max_epoch=args.client_local_epoch,
    validate_ratio=0,
    save_model=False,
)

weights_2 = client_model.get_weights()['trainable']

gradients = {key: (weights_1[key] - weights_2[key]) / client_lr / args.client_local_epoch for key in weights_1}

attack_gradients = gradients['dense/kernel:0']

for i in range(len(x_train)):
    plt.imsave(os.path.join(image_dir, 'RealImage-%s.png' % (i)),
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
    print('Image:', i, 'L2 Distance:', np.min(error_all[:, i]))
    plt.imsave(os.path.join(image_dir, 'attack-%s.png' % i), img_list[attack_index].reshape([28, 28]), cmap='gray')
    result_img_list.append(img_list[attack_index].reshape([28, 28]))

with open('FC_Attack_Results.txt', 'a+') as f:
    f.write(', '.join([str(args.training_samples), str(args.client_batch_size), str(args.client_local_epoch),
                       str(error_all.min(0).mean())]) + '\n')

plot_image = [x_train, result_img_list]

Nr, Nc = 2, len(x_train)
fig, axs = plt.subplots(Nr, Nc, figsize=(Nc, Nr))

for i in range(Nr):
    for j in range(Nc):
        axs[i, j].axis('off')
        axs[i, j].imshow(plot_image[i][j].reshape([28, 28]), cmap='gray')

fig.tight_layout()
plt.savefig(os.path.join(image_dir, 'attack-result.png'), type="png", dpi=300)

print('Attach finish, please view the results in:', image_dir)