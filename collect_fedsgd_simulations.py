import os
import numpy as np

log_dir = 'log/tune_lr_jmlr/fed_sgd_simulator'

log_files = [os.path.join(log_dir, e) for e in os.listdir(log_dir) if e.endswith('fed_sgd_simulator.csv')]

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

