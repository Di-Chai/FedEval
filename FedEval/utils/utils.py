import codecs
import multiprocessing
import os
import pickle
import shutil
import time

import numpy as np
from scipy.stats import wasserstein_distance
from .multi_threads import multiple_process
from numpy.lib.format import open_memmap


def obj_to_pickle_string(x, file_path=None):
    if file_path is not None:
        with open(file_path, 'wb') as output:
            pickle.dump(x, output, protocol=4)
        return file_path
    else:
        return codecs.encode(pickle.dumps(x, protocol=4), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    if ".pkl" in s:
        df = open(s, "rb")
        return pickle.load(df)
    else:
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


def get_data_hist(records):
    user_data, hist_min, hist_max = records
    user_data = np.reshape(user_data, [len(user_data), -1])
    tmp_hists = []
    for i in range(user_data.shape[-1]):
        n, bins = np.histogram(
            user_data[:, i], bins=10, range=(hist_min, hist_max), density=True)
        tmp_hists.append(n.astype(np.float32))
    return tmp_hists


# def emd_element(record):
#     i, j, uh_i, uh_j = record
#     tmp = []
#     for k in range(len(uh_i)):
#         tmp.append(wasserstein_distance(
#             uh_i[k][0], uh_j[k][0],
#             uh_i[k][1][1:], uh_j[k][1][1:]
#         ))
#     return i, j, np.mean(tmp)


def emd_element(share_queue, locker, data, parameter_list):

    print(f'Child Process {os.getpid()} Running')

    user_hist = open_memmap(parameter_list[1])
    hist_min, hist_max = parameter_list[2], parameter_list[3]
    bins = (hist_max - hist_min) / 10 * (np.array(list(range(1, 11)))) + hist_min
    emd_matrix = np.zeros([len(user_hist), len(user_hist)])
    counter = 0
    for i, j in data:
        tmp = []
        for k in range(len(user_hist[i])):
            try:
                tmp.append(wasserstein_distance(
                    user_hist[i][k], user_hist[j][k],
                    bins, bins
                ))
            except Exception as e:
                print(e, user_hist[i][k], user_hist[j][k], bins)
                exit(1)
        emd_matrix[i][j] = np.mean(tmp)
        counter += 1

    locker.acquire()
    share_queue.put(emd_matrix)
    locker.release()


def get_emd(non_iid_data):

    n_job = multiprocessing.cpu_count()

    hist_min = min([e['x_train'].min() for e in non_iid_data])
    hist_max = max([e['x_train'].max() for e in non_iid_data])

    if hist_min < 0:
        for i in range(len(non_iid_data)):
            non_iid_data[i]['x_train'] -= hist_min
        hist_max -= hist_min
        hist_min -= hist_min

    with multiprocessing.Pool(n_job) as p:
        user_hists = p.map(get_data_hist, [(e['x_train'], hist_min, hist_max) for e in non_iid_data])

    user_hists = np.array(user_hists, dtype=np.float32)

    tmp_file_name = 'tmp_user_hist.npy'
    np.save(tmp_file_name, user_hists)

    del non_iid_data

    target_index = []
    for i in range(len(user_hists)):
        for j in range(len(user_hists)):
            if i > j:
                continue
            target_index.append([i, j])

    emd_matrix = multiple_process(
        distribute_list=target_index,
        partition_data_func=lambda data, i, n_job: [data[e] for e in range(len(data)) if e % n_job == i],
        task_function=emd_element,
        n_jobs=n_job, reduce_function=lambda x, y: x + y, parameter_list=[tmp_file_name, hist_min, hist_max]
    )

    emd_matrix += emd_matrix.T

    shutil.rmtree(tmp_file_name, ignore_errors=True)

    return np.mean(emd_matrix)


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
