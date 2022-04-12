import os
import numpy as np

log_dir = 'log/tunelr_jmlr/central_simulator'

log_files = [os.path.join(log_dir, e) for e in os.listdir(log_dir) if e.endswith('central_simulator.csv')]

result_dict = {}
for log_file in log_files:

    with open(log_file, 'r') as f:
        central_simulation = f.readlines()

    if not central_simulation[-1].startswith('Best TEST Metric'):
        print('Incomplete log file, Skip')
        continue

    val_acc = central_simulation[-2].strip('\n').split(', ')[-1]
    test_acc = central_simulation[-1].strip('\n').split(', ')[-1]

    result_dict[central_simulation[-3]] = result_dict.get(central_simulation[-3], []) + [[
        float(val_acc), float(test_acc)]]

    print(central_simulation[-3].strip('\n'), val_acc, test_acc, 'max round', central_simulation[-4].split(',')[0])

results = []
for key in result_dict:
    acc_mean = ', '.join(np.mean(result_dict[key], axis=0).astype(str).tolist())
    acc_std = ', '.join(np.std(result_dict[key], axis=0).astype(str).tolist())
    results.append(
        key.strip('\n') + ', ' + acc_mean + ', ' + acc_std + '\n'
    )
    print('Repeat', key.strip('\n'), len(result_dict[key]))

with open('simulate_local.csv', 'w') as f:
    f.write('Dataset, #Clients, LR, ValAcc, TestAcc, ValStd, TestStd\n')
    f.writelines(results)

