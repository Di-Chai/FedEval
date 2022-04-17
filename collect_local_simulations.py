import os
import shutil
import numpy as np

log_dir = 'log/tune_lr_jmlr/local_simulator'

log_files = [os.path.join(log_dir, e) for e in os.listdir(log_dir) if e.endswith('local_simulator.csv')]

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
    print('Repeat', key.strip('\n'), len(result_dict[key]))

with open('simulate_local.csv', 'w') as f:
    f.write('Repeat, Dataset, #Clients, LR, TestAcc, TestStd\n')
    f.writelines(results)
