from .FedImage import *
from .Semantic140 import *
from .Shakespeare import *


# Used by the server, because it cannot reach the raw data
def get_data_shape(dataset):
    if dataset == 'celeba':
        x_size = (None, 54, 44, 3)
        y_size = (None, 2)
    elif dataset == 'femnist':
        x_size = (None, 28, 28, 1)
        y_size = (None, 62)
    elif dataset == 'mnist':
        x_size = (None, 28, 28, 1)
        y_size = (None, 10)
    elif dataset == 'cifar10':
        x_size = (None, 32, 32, 3)
        y_size = (None, 10)
    elif dataset == 'cifar100':
        x_size = (None, 32, 32, 3)
        y_size = (None, 100)
    elif dataset == 'shakespeare':
        x_size = (None, 80)
        y_size = (None, 80)
    elif dataset == 'semantic140':
        x_size = (None, 25, 200)
        y_size = (None, 1)
    else:
        raise ValueError('Unknown dataset', dataset)
    return x_size, y_size