import os

datasets = ['mnist']

models = ['MLP', 'LeNet']

upload_sparse = 1.0
upload_strategy = 'no-compress'

non_iid_list = [
    # [0, 'iid'],
    [1, 'average'],
    [2, 'average'],
    [3, 'average'],
]

num_clients = 100
max_epochs = 2000

optimizer = 'adam'
upload_name_filter = 'None'

B_C_E_LR = [

    # Repeat 10 times, and compute the averaged results

    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],
    [1000, 1, 1, 1e-2],

    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],
    [8, 0.1, 16, 5e-4],

]

order = 'python run_base_local.py --dataset {} --model {} --non-iid {} --non-iid-strategy {} ' \
        '--B {} --C {} --E {} --file_name {} --num_clients {} --max_epochs {} --lr {} --optimizer {} ' \
        '--upload_name_filter {} --upload_sparse {} --upload_strategy {} ' \
        '--sudo No'

file_name = os.path.basename(__file__).strip('run_ .py') + '_trials.txt'

for dataset in datasets:
    for model in models:
        for b, c, e, lr in B_C_E_LR:
            for non_iid, non_iid_strategy in non_iid_list:
                os.system(order.format(dataset, model, non_iid, non_iid_strategy, b, c, e, file_name,
                                       num_clients, max_epochs, lr, optimizer, upload_name_filter,
                                       upload_sparse, upload_strategy))
