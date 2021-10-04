import codecs
import os
import pickle

import numpy as np


def obj_to_pickle_string(x, file_path=None):
    if file_path is not None:
        with open(file_path, 'wb') as output:
            pickle.dump(x, output)
        return file_path
    else:
        return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    if ".pkl" in s:
        df = open(s, "rb")
        print("load model from file")
        return pickle.load(df)
    else:
        print("load model from byte")
        return pickle.loads(codecs.decode(s.encode(), "base64"))


def list_tree(value_lists):
    result = []
    counter = np.zeros(len(value_lists), dtype=np.int32)
    for i in range(int(np.prod([len(e) for e in value_lists]))):
        tmp = []
        for j in range(len(value_lists)):
            v = value_lists[j][counter[j]]
            if type(v) is list:
                tmp += v
            else:
                tmp.append(v)
        result.append(tmp)
        counter[-1] += 1
        for j in range(len(value_lists)-1, -1, -1):
            if counter[j] > (len(value_lists[j]) - 1):
                counter[j] = 0
                if j > 0:
                    counter[j-1] += 1
    return result


# def tune_lr_bce(lrs=(5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2),
#                 B=(32, 16, 8, 4, 2, 1), C=(0.1,), E=(2, 4, 8, 16, 32, 64)):
#     params = []
#     for lr in lrs:
#         for b in B:
#             for c in C:
#                 for e in E:
#                     params.append({'lr': lr, 'B': b, 'C': c, 'E': e})
#     return params
#
#
# def get_params_dict():
#     return {
#         'exec': 'run',
#         'mode': 'server',
#         'file': 'configs,tf_wrapper',
#         'config': None,
#         'dataset': 'mnist',
#         'ml_model': 'LeNet',
#         'fed_model': 'FedSGD',
#         'optimizer': 'sgd',
#         'upload_optimizer': 'False',
#         'upload_sparsity': 1.0,
#         'upload_dismiss': 'None',
#         'lazy_update': 'False',
#         'B': 1000,
#         'C': 1.0,
#         'E': 1,
#         'num_tolerance': 100,
#         'num_clients': 100,
#         'max_epoch': 5000,
#         'non-iid': 0,
#         'non-iid-strategy': 'iid',
#         'lr': None,
#         'output': 'results.txt',
#         'sudo': 'True'
#     }
#
#
# def grid_search(
#         execution, dataset, mode, config, ml_model, fed_model, output,
#         repeat=1,
#         tune_B=(32, 16, 8, 4, 2, 1),
#         tune_C=(0.1,),
#         tune_E=(2, 4, 8, 16, 32, 64),
#         tune_LR=(5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2),
#         **kwargs
# ):
#     trial_params = {'dataset': dataset, 'ml_model': ml_model, 'fed_model': fed_model,
#                     'mode': mode, 'exec': execution, 'config': config, 'output': output}
#     default_params = get_params_dict()
#     params_tuning = tune_lr_bce(B=tune_B, C=tune_C, E=tune_E, lrs=tune_LR) * repeat
#
#     default_params.update(trial_params)
#     default_params.update(kwargs)
#
#     if execution == 'run':
#         for p in params_tuning:
#             default_params.update(p)
#             os.system('python -m FedEval.run_util ' +
#                       ' '.join(["--%s %s" % (key, value) for key, value in default_params.items()]))
#
#     if execution == 'stop':
#         os.system('python -m FedEval.run_util --mode {} --config {} --exec stop'.format(mode, config))
