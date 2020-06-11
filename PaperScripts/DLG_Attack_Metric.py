import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_match(RI, FI):
    distance = []
    for real in RI:
        tmp = []
        for fake in FI:
            tmp.append(np.mean(np.square(real - fake)))
        distance.append(tmp)
    return np.array(distance, dtype=np.float32)


def attack_evaluation(mode, num_images, E, repeat_index=0):

    dir_name = "{}-I{}-E{}-{}".format(mode, num_images, E, repeat_index)

    image_path = os.path.join('image', dir_name)

    images = [e for e in os.listdir(image_path) if e.endswith('.png')]

    real_image_names = [e for e in images if e.startswith('real_image')]
    real_labels = [int(e.strip('.png').split('_')[-1]) for e in real_image_names]

    fake_image_names = [e for e in images if e.startswith('fake_image')]
    fake_image_names = sorted(fake_image_names, key=lambda x: int(x.split('_')[2]), reverse=True)[:len(real_image_names)]
    fake_labels = [int(e.strip('.png').split('_')[-1]) for e in fake_image_names]

    # print('Real Images', real_image_names)
    # print('Fake Images', fake_image_names)

    real_images = []
    for name in real_image_names:
        real_images.append(cv2.imread(os.path.join(image_path, name), cv2.IMREAD_GRAYSCALE) / 255)

    fake_images = []
    for name in fake_image_names:
        fake_images.append(cv2.imread(os.path.join(image_path, name), cv2.IMREAD_GRAYSCALE) / 255)

    match_matrix = image_match(fake_images, real_images)

    match_id = np.argmin(match_matrix, axis=1)
    match_error = match_matrix.min(1).mean()

    label_acc = []
    for i in range(len(fake_image_names)):
        real_label = int(real_labels[match_id[i]])
        fake_label = int(fake_labels[i])
        label_acc.append(1 if real_label == fake_label else 0)

    # print('Label Acc', np.mean(label_acc))
    # print('Loss', match_error)

    return np.mean(label_acc), match_error


if __name__ == '__main__':

    #######################################################################################
    # Varying # of images, DLG, FedSGD, LeNet
    mode = 'DLG-LeNet-FedSGD'
    num_images = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_epochs = [1]
    repeat_times = 50

    for ni in num_images:
        for ne in num_epochs:
            tmp_acc, tmp_loss = [], []
            for repeat_index in range(repeat_times):
                acc, loss = attack_evaluation(mode, ni, ne, repeat_index)
                tmp_acc.append(acc)
                tmp_loss.append(loss)
            print(mode, '#Images=%s' % ni, '#Epochs=%s' % ne,
                  'LabelAcc=%.3f' % np.mean(tmp_acc), 'L2-Distance=%s' % np.mean(tmp_loss))

    #######################################################################################
    # Varying # of images, DLG, FedAvg, LeNet
    mode = 'DLG-LeNet-FedAvg'
    num_images = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    num_epochs = [20]
    repeat_times = 50

    for ni in num_images:
        for ne in num_epochs:
            tmp_acc, tmp_loss = [], []
            for repeat_index in range(repeat_times):
                acc, loss = attack_evaluation(mode, ni, ne, repeat_index)
                tmp_acc.append(acc)
                tmp_loss.append(loss)
            print(mode, '#Images=%s' % ni, '#Epochs=%s' % ne,
                  'LabelAcc=%.3f' % np.mean(tmp_acc), 'L2-Distance=%s' % np.mean(tmp_loss))

    #######################################################################################
    # Varying # of epochs, DLG, FedAvg, LeNet
    mode = 'DLG-LeNet-FedAvg'
    num_images = [2, 6]
    num_epochs = [1, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    repeat_times = 50

    for ni in num_images:
        for ne in num_epochs:
            tmp_acc, tmp_loss = [], []
            for repeat_index in range(repeat_times):
                acc, loss = attack_evaluation(mode, ni, ne, repeat_index)
                tmp_acc.append(acc)
                tmp_loss.append(loss)
            print(mode, '#Images=%s' % ni, '#Epochs=%s' % ne,
                  'LabelAcc=%.3f' % np.mean(tmp_acc), 'L2-Distance=%s' % np.mean(tmp_loss))

    #######################################################################################
    # Varying # of images, FedSGD, MLP
    mode = 'DLG-MLP-FedSGD'
    num_images = [1, 5, 10, 15, 0, 25, 30]
    num_epochs = [1]
    repeat_times = 50

    for ni in num_images:
        for ne in num_epochs:
            tmp_acc, tmp_loss = [], []
            for repeat_index in range(repeat_times):
                acc, loss = attack_evaluation(mode, ni, ne, repeat_index)
                tmp_acc.append(acc)
                tmp_loss.append(loss)
            print(mode, '#Images=%s' % ni, '#Epochs=%s' % ne,
                  'LabelAcc=%.3f' % np.mean(tmp_acc), 'L2-Distance=%s' % np.mean(tmp_loss))

    #######################################################################################
    # Varying # of images, FedAvg, MLP
    mode = 'DLG-MLP-FedAvg'
    num_images = [1, 5, 10, 15, 20, 25, 30]
    num_epochs = [20]
    repeat_times = 50

    for ni in num_images:
        for ne in num_epochs:
            tmp_acc, tmp_loss = [], []
            for repeat_index in range(repeat_times):
                acc, loss = attack_evaluation(mode, ni, ne, repeat_index)
                tmp_acc.append(acc)
                tmp_loss.append(loss)
            print(mode, '#Images=%s' % ni, '#Epochs=%s' % ne,
                  'LabelAcc=%.3f' % np.mean(tmp_acc), 'L2-Distance=%s' % np.mean(tmp_loss))

    #######################################################################################
    # Varying # of epochs, FedAvg, MLP
    mode = 'DLG-MLP-FedAvg'
    num_images = [10, 20, 30]
    num_epochs = [1, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    repeat_times = 50

    for ni in num_images:
        for ne in num_epochs:
            tmp_acc, tmp_loss = [], []
            for repeat_index in range(repeat_times):
                acc, loss = attack_evaluation(mode, ni, ne, repeat_index)
                tmp_acc.append(acc)
                tmp_loss.append(loss)
            print(mode, '#Images=%s' % ni, '#Epochs=%s' % ne,
                  'LabelAcc=%.3f' % np.mean(tmp_acc), 'L2-Distance=%s' % np.mean(tmp_loss))
