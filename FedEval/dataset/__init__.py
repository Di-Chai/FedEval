from .FedImage import *
from .FedMatrix import *
from .Semantic140 import *
from .Shakespeare import *


# Used by the server, because it cannot reach the raw data
# Q(fgh): shouldn't this be returned by the dataset object respectively?
def get_data_shape(dataset_name: str):
    if dataset_name == 'celeba':
        x_size = (None, 54, 44, 3)
        y_size = (None, 2)
    elif dataset_name == 'femnist':
        x_size = (None, 28, 28, 1)
        y_size = (None, 62)
    elif dataset_name == 'mnist':
        x_size = (None, 28, 28, 1)
        y_size = (None, 10)
    elif dataset_name == 'cifar10':
        x_size = (None, 32, 32, 3)
        y_size = (None, 10)
    elif dataset_name == 'cifar100':
        x_size = (None, 32, 32, 3)
        y_size = (None, 100)
    elif dataset_name == 'shakespeare':
        x_size = (None, 80)
        y_size = (None, 80)
    elif dataset_name == 'semantic140':
        x_size = (None, 25, 200)
        y_size = (None, 1)
    elif dataset_name == 'wine':
        x_size = (6480, 12)
        y_size = (6480, 2)
    elif dataset_name == 'mnist_matrix':
        x_size = (60000, 784)
        y_size = (60000, 10)
    elif dataset_name == 'synthetic_matrix_horizontal':
        x_size = (None, None)
        y_size = (None, 1)
    elif dataset_name == 'synthetic_matrix_vertical':
        x_size = (None, None)
        y_size = (None, 1)
    elif dataset_name == 'vertical_linear_regression':
        x_size = (None, None)
        y_size = (None, 1)
    elif dataset_name == 'synthetic_matrix_horizontal_memmap':
        x_size = (None, None)
        y_size = (None, 1)
    else:
        raise ValueError('Unknown dataset', dataset_name)
    return x_size, y_size