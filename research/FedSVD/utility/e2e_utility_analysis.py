import os
import copy
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import TruncatedSVD, PCA
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, NullFormatter
import matplotlib.ticker as tck


figure_path = os.path.join('.', 'figure')
if not os.path.isdir(figure_path):
    os.makedirs(figure_path)

plt.rcParams.update({
    'font.size': 10, "font.family": "Times New Roman",
    'font.weight': 'bold',
})
colors = ['red', 'orange', 'yellow', 'green', 'gold', 'blue', 'aqua', 'black', 'purple', 'navy']
pad_inches = 0.1


def pca(x, k):
    # Zero Mean
    x -= np.mean(x, axis=0)
    svd = TruncatedSVD(n_components=k, algorithm='arpack')
    svd.fit(x)
    # Return the transformation matrix
    return svd.components_


def rmse(x, x_):
    return np.sqrt(np.mean(np.square(x - x_)))


def mape(x, x_):
    return np.mean(np.abs(x - x_) / np.abs(x))


def projection_distance(x, x_):
    return rmse(x @ x.T, x_ @ x_.T)


# Prepare the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.reshape(x_train, [x_train.shape[0], -1]).astype(np.float64)
x_test = np.reshape(x_test, [x_test.shape[0], -1]).astype(np.float64)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

x_train = x_train[np.logical_or(y_train == 5, y_train == 1)]
y_train = y_train[np.logical_or(y_train == 5, y_train == 1)]

# PCA and Transform
num_sample_list = [1000, 5000, 10000, 15000, 20000]
transform_results = []
for num_samples in num_sample_list:
    pca_x = copy.deepcopy(x_train[:num_samples])
    pca_map = pca(pca_x, k=2)
    t_x = pca_x @ pca_map.T
    transform_results.append(t_x[:1000])
    print(num_samples, rmse(pca_x[:1000], (pca_x @ pca_map.T @ pca_map)[:1000]))


# Plot
for i in range(len(transform_results)):
    t_r = transform_results[i]
    fig = figure(figsize=(2, 2), dpi=200)
    plt.scatter(
        [e[0] for e in t_r], [e[1] for e in t_r], c=[colors[y_train[e]] for e in range(len(t_r))],
        s=2.0
    )
    plt.xlabel('1st PC', fontweight='bold')
    plt.ylabel('2nd PC', fontweight='bold')
    # plt.title(f'#Sample={num_sample_list[i]}', fontweight='bold')
    fig.savefig(
        os.path.join(figure_path, f'pca_{num_sample_list[i]}.png'),
        bbox_inches='tight', pad_inches=pad_inches
    )
    plt.close()

for i in range(50):
    fig = figure(figsize=(1, 1), dpi=200)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    fig.savefig(
        os.path.join(figure_path, f'image_sample_{i}_{y_train[i]}.png'),
        bbox_inches='tight', pad_inches=pad_inches
    )
    plt.close()
