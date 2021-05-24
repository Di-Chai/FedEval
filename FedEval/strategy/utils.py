import numpy as np


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
