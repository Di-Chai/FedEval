import os
import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from matplotlib import patches

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

###########################################################################################
# Train the CNN model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

raw_shape = x_train.shape[1:]

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train /= 255
x_test /= 255

model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

if os.path.isfile('FC_Attack_Model.pkl'):
    model.load_weights('FC_Attack_Model.pkl')
else:
    model.fit(x=x_train, y=y_train, batch_size=64, epochs=50,
              verbose=2, validation_split=0.1)
    model.save('FC_Attack_Model.pkl')


def print_results(mode, num_images, num_epochs, repeats):
    for ni in num_images:
        for ne in num_epochs:
            tmp_acc = []
            tmp_l2 = []
            for repeat_index in range(repeats):
                image_path = os.path.join('image', '{}-I{}-E{}-{}'.format(mode, ni, ne, repeat_index))
                real_images = []
                for i in range(ni):
                    real_images.append(cv2.imread(os.path.join(image_path, 'real-%s.png' % i),
                                                  cv2.IMREAD_GRAYSCALE) / 255)
                attack_images = []
                for i in range(ni):
                    attack_images.append(cv2.imread(os.path.join(image_path, 'attack-%s.png' % i),
                                                    cv2.IMREAD_GRAYSCALE) / 255)

                real_labels = model.predict(x=np.expand_dims(np.array(real_images), -1))
                attack_labels = model.predict(x=np.expand_dims(np.array(attack_images), -1))

                label_acc = np.mean(np.argmax(real_labels, 1) == np.argmax(attack_labels, 1))
                l2_distance = np.square(np.array(real_images) - np.array(attack_images)).mean()

                tmp_acc.append(label_acc)
                tmp_l2.append(l2_distance)

                plot_image = [real_images, attack_images]

                Nr, Nc = 2, len(real_images)
                fig, axs = plt.subplots(Nr, Nc, figsize=(Nc, Nr))

                if Nc == 1:
                    for i in range(Nr):
                        try:
                            axs[i].axis('off')
                            axs[i].set_xlabel('test')
                            axs[i].imshow(plot_image[i][0].reshape([28, 28]), cmap='gray')
                            if np.argmax(real_labels[0]) != np.argmax(attack_labels[0]):
                                rect = patches.Rectangle((0, 0), 27, 27, linewidth=4, edgecolor='r', facecolor='none')
                                axs[i].add_patch(rect)
                        except Exception as e:
                            print(i, 0, e)
                else:
                    for i in range(Nr):
                        for j in range(Nc):
                            try:
                                axs[i, j].axis('off')
                                axs[i, j].imshow(plot_image[i][j].reshape([28, 28]), cmap='gray')
                                if np.argmax(real_labels[j]) != np.argmax(attack_labels[j]):
                                    rect = patches.Rectangle((0, 0), 27, 27, linewidth=4, edgecolor='r', facecolor='none')
                                    axs[i, j].add_patch(rect)
                            except Exception as e:
                                print(i, j, e)
                fig.tight_layout()
                plt.savefig(os.path.join(image_path, 'attack-result-mark.png'), type="png", dpi=300)

            print(mode, '#Images=%s' % ni, '#Epochs=%s' % ne,
                  'LabelAcc=%.3f' % np.mean(tmp_acc), 'L2-Distance=%s' % np.mean(tmp_l2))


if __name__ == '__main__':

    ############################################################################
    # Varying # of images, FC, FedSGD, MLP
    mode = 'FC-MLP-FedSGD'
    num_images = [1, 5, 10, 15, 20, 25, 30]
    num_epochs = [1]
    repeat_times = 10

    print_results(mode=mode, num_images=num_images, num_epochs=num_epochs, repeats=repeat_times)

    ############################################################################
    # Varying # of images, FC, FedAvg, MLP
    mode = 'FC-MLP-FedAvg'
    num_images = [1, 5, 10, 15, 20, 25, 30]
    num_epochs = [20]
    repeat_times = 10

    print_results(mode=mode, num_images=num_images, num_epochs=num_epochs, repeats=repeat_times)

    ############################################################################
    # Varying # of epochs, FC, FedAvg, MLP
    mode = 'FC-MLP-FedAvg'
    num_images = [10, 20, 30]
    num_epochs = [1, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    repeat_times = 10

    print_results(mode=mode, num_images=num_images, num_epochs=num_epochs, repeats=repeat_times)