import os
import numpy as np


def process_central_simulate_results(log_files):
    result_dict = {}
    for log_file in log_files:

        with open(log_file, 'r') as f:
            central_simulation = f.readlines()

        if not central_simulation[-1].startswith('Best TEST Metric'):
            print('Incomplete log file, Skip')
            os.remove(log_file)
            continue

        duration = central_simulation[-3].strip('\n').split(' ')[-1]
        val_acc = central_simulation[-2].strip('\n').split(', ')[-1]
        test_acc = central_simulation[-1].strip('\n').split(', ')[-1]

        result_dict[central_simulation[-4]] = result_dict.get(central_simulation[-4], []) + [[
            float(duration), float(val_acc), float(test_acc)]]

        print(central_simulation[-4].strip('\n'), val_acc, test_acc, 'max round', central_simulation[-5].split(',')[0])

    results = []
    for key in result_dict:
        acc_mean = ', '.join(np.mean(result_dict[key], axis=0).astype(str).tolist())
        acc_std = ', '.join(np.std(result_dict[key], axis=0).astype(str).tolist())
        results.append(
            str(len(result_dict[key])) + ', ' + key.strip('\n') + ', ' + acc_mean + ', ' + acc_std + '\n'
        )
        print('Repeat', key.strip('\n'), len(result_dict[key]))

    with open('simulate_central.csv', 'w') as f:
        f.write('Repeat, Dataset, #Clients, LR, Duration, ValAcc, TestAcc, DurationStd, ValStd, TestStd\n')
        f.writelines(results)


def process_fedsgd_simulate_results(log_files):
    result_dict = {}
    for log_file in log_files:

        with open(log_file, 'r') as f:
            fed_sgd_simulation = f.readlines()

        if not fed_sgd_simulation[-1].startswith('Best Metric'):
            print('Incomplete log file, Skip')
            os.remove(log_file)
            continue

        test_acc = fed_sgd_simulation[-1].strip('\n').split(', ')[-1]

        result_dict[fed_sgd_simulation[0]] = result_dict.get(fed_sgd_simulation[0], []) + [[
            int(fed_sgd_simulation[-2].split(',')[0]), float(test_acc)]]

        print(fed_sgd_simulation[0].strip('\n'), test_acc, 'max round', fed_sgd_simulation[-2].split(',')[0])

    results = []
    for key in result_dict:
        acc_mean = ', '.join(np.mean(result_dict[key], axis=0).astype(str).tolist())
        acc_std = ', '.join(np.std(result_dict[key], axis=0).astype(str).tolist())
        results.append(
            str(len(result_dict[key])) + ', ' + key.strip('\n') + ', ' + acc_mean + ', ' + acc_std + '\n'
        )
        print('Repeat', key.strip('\n'), len(result_dict[key]))

    with open('simulate_fed_sgd.csv', 'w') as f:
        f.write('Repeat, Dataset, #Clients, LR, Round, TestAcc, RoundStd, TestStd\n')
        f.writelines(results)


def process_local_simulate_results(log_files):
    result_dict = {}
    for log_file in log_files:

        with open(log_file, 'r') as f:
            local_simulation = f.readlines()

        if not local_simulation[-1].startswith('Average Best Test Metric'):
            print('Incomplete log file, Skip')
            os.remove(log_file)
            continue

        test_acc = local_simulation[-1].strip('\n').split(', ')[-1]

        result_dict[local_simulation[-5]] = result_dict.get(local_simulation[-5], []) + [[float(test_acc)]]

        print(local_simulation[-5].strip('\n'), test_acc)

    results = []
    for key in result_dict:
        acc_mean = ', '.join(np.mean(result_dict[key], axis=0).astype(str).tolist())
        acc_std = ', '.join(np.std(result_dict[key], axis=0).astype(str).tolist())
        results.append(
            str(len(result_dict[key])) + ', ' + key.strip('\n') + ', ' + acc_mean + ', ' + acc_std + '\n'
        )

    with open('simulate_local.csv', 'w') as f:
        f.write('Repeat, Dataset, #Clients, LR, TestAcc, TestStd\n')
        f.writelines(results)
        print('Repeat', key.strip('\n'), len(result_dict[key]))


if __name__ == '__main__':

    log_dir = 'log/JMLR/'
    log_dir = [os.path.join(log_dir, e) for e in os.listdir(log_dir) if e.startswith('2022')]

    log_files_central_simulate = []
    log_files_local_simulate = []
    log_files_fedsgd_simulate = []
    for ld in log_dir:
        files = [os.path.join(ld, e) for e in os.listdir(ld)]
        log_files_central_simulate += [e for e in files if e.endswith('central_simulator.csv')]
        log_files_fedsgd_simulate += [e for e in files if e.endswith('fed_sgd_simulator.csv')]
        log_files_local_simulate += [e for e in files if e.endswith('local_simulator.csv')]

    if len(log_files_central_simulate) > 0:
        process_central_simulate_results(log_files_central_simulate)
    if len(log_files_fedsgd_simulate) > 0:
        process_fedsgd_simulate_results(log_files_fedsgd_simulate)
    if len(log_files_local_simulate) > 0:
        process_local_simulate_results(log_files_local_simulate)
