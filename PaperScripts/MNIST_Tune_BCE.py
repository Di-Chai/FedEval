import os

datasets = ['mnist']

models = ['MLP', 'LeNet']

non_iid_list = [
    [0, 'iid'],
]

num_clients = 100
max_epochs = 2000
lr_list = [5e-4]

optimizers = ['Adam']
upload_name_filter = 'None'

compress = [
    # upload_sparse, upload_strategy
    [1.0, 'no-compress']
]

B_C_E = [

    # FedAvg
    [32, 0.1, 2],
    [32, 0.1, 4],
    [32, 0.1, 8],
    [32, 0.1, 16],
    [32, 0.1, 32],
    [32, 0.1, 64],
    [16, 0.1, 2],
    [16, 0.1, 4],
    [16, 0.1, 8],
    [16, 0.1, 16],
    [16, 0.1, 32],
    [16, 0.1, 64],
    [8, 0.1, 2],
    [8, 0.1, 4],
    [8, 0.1, 8],
    [8, 0.1, 16],
    [8, 0.1, 32],
    [8, 0.1, 64],
    [4, 0.1, 2],
    [4, 0.1, 4],
    [4, 0.1, 8],
    [4, 0.1, 16],
    [4, 0.1, 32],
    [4, 0.1, 64],
    [2, 0.1, 2],
    [2, 0.1, 4],
    [2, 0.1, 8],
    [2, 0.1, 16],
    [2, 0.1, 32],
    [2, 0.1, 64],
]

order = 'python run_base_local.py --dataset {} --model {} --non-iid {} --non-iid-strategy {} ' \
        '--B {} --C {} --E {} --file_name {} --num_clients {} --max_epochs {} --lr {} --optimizer {} ' \
        '--upload_name_filter {} --upload_sparse {} --upload_strategy {} ' \
        '--sudo no'

file_name = os.path.basename(__file__).strip('run_ .py') + '_results.txt'

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