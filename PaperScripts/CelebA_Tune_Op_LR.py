import os

datasets = ['celeba']

models = ['LeNet']

non_iid_list = [
    [0, 'iid'],
]

num_clients = 100
max_epochs = 2000

lr_list = [
    1e-4, 5e-4, 1e-3, 5e-3, 1e-2
]

optimizers = ['Adam', 'GD', 'Momentum']
upload_name_filter = 'None'

compress = [
    # upload_sparse, upload_strategy
    [1.0, 'no-compress'],
]

B_C_E = [
    # FedSGD
    [1000, 1, 1],
    # FedAvg
    [8, 0.1, 16],
]

order = 'python run_base_local.py --dataset {} --model {} --non-iid {} --non-iid-strategy {} ' \
        '--B {} --C {} --E {} --file_name {} --num_clients {} --max_epochs {} --lr {} --optimizer {} ' \
        '--upload_name_filter {} --upload_sparse {} --upload_strategy {} ' \
        '--sudo no'

file_name = os.path.basename(__file__).strip('run_ .py') + '_results.txt'

if os.path.isfile(file_name):
    with open(file_name, 'r') as f:
        results = f.readlines()

for comp in compress:
    for dataset in datasets:
        for optimizer in optimizers:
            for lr in lr_list:
                for model in models:
                    for non_iid, non_iid_strategy in non_iid_list:
                        for b, c, e in B_C_E:
                            os.system(order.format(dataset, model, non_iid, non_iid_strategy, b, c, e, file_name,
                                                   num_clients, max_epochs, lr, optimizer, upload_name_filter,
                                                   comp[0], comp[1]))
