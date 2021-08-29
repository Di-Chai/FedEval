import os

import tensorflow as tf

from ..utils import obj_to_pickle_string, pickle_string_to_obj


def aggregate_weighted_average(client_params, aggregate_weights):
    """
    Args:
        client_params: [params1, params2, ...] are the weights form different clients
        aggregate_weights: aggregate weights of different clients, usually set according to the
            clients' training samples. E.g., A, B, and C have 10, 20, and 30 images, then the
            aggregate_weights = [1/6, 1/3, 1/2]

    Returns: the aggregated parameters, which have the same format with any instance from the
        client_params
    """
    new_param = []
    for i in range(len(client_params[0])):
        for j in range(len(client_params)):
            if j == 0:
                new_param.append(client_params[j][i] * aggregate_weights[j])
            else:
                new_param[i] += client_params[j][i] * aggregate_weights[j]
    return new_param


def save_fed_model(fed_model, path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    # Save ML-Model Weights
    ml_model = fed_model.ml_model
    ml_model.save_weights(os.path.join(path, 'ml_model.h5'), save_format='h5')
    fed_model.ml_model = None
    # Save the data
    data = {}
    data_keys = ['train_data', 'val_data', 'test_data']
    for key in data_keys:
        if hasattr(fed_model, key):
            data[key] = getattr(fed_model, key)
            setattr(fed_model, key, None)
    if len(data) > 0 and os.path.isfile(os.path.join(path, 'data.pkl')) is False:
        obj_to_pickle_string(data, os.path.join(path, 'data.pkl'))
    # Save the fed model
    obj_to_pickle_string(fed_model, os.path.join(path, 'fed_model.pkl'))
    # restore the model and data
    fed_model.ml_model = ml_model
    if len(data) > 0:
        for key in data_keys:
            setattr(fed_model, key, data[key])
    return fed_model


def load_fed_model(fed_model, path):
    new_fed_model = pickle_string_to_obj(os.path.join(path, 'fed_model.pkl'))
    new_fed_model.ml_model = fed_model.ml_model
    new_fed_model.ml_model.load_weights(os.path.join(path, 'ml_model.h5'))
    data = pickle_string_to_obj(os.path.join(path, 'data.pkl'))
    data_keys = ['train_data', 'val_data', 'test_data']
    for key in data_keys:
        setattr(new_fed_model, key, data[key])
    return new_fed_model
