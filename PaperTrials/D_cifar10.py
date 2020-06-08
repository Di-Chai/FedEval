import os

os.chdir('../')

params_run = [

    # dataset, model, optimizer, upload_name_filter, lr, non_iid, non_iid_strategy, b, c, e

    ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 1000, 1.0, 1],

    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 1.0, 1],
    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 0.8, 1],
    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 0.6, 1],
    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 0.4, 4],
    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 0.4, 16],
    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 0.4, 32],
    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 0.2, 1],
    # ['cifar10', 'MobileNet', 'adam', 'None', 5e-4, 0, 'iid', 32, 0.1, 1],
]

num_clients = 100
max_epochs = 2000

upload_sparse = 1.0
upload_strategy = 'no-compress'

order = 'python3 run_base_server.py --dataset {} --model {} --non-iid {} --non-iid-strategy {} ' \
        '--B {} --C {} --E {} --file_name {} --num_clients {} --max_epochs {} --lr {} --optimizer {} ' \
        '--upload_name_filter {} --upload_sparse {} --upload_strategy {} ' \
        '--sudo sudo --path configs/cluster3'

file_name = os.path.basename(__file__).strip('run_ .py') + '_trials.txt'

for dataset, model, optimizer, upload_name_filter, lr, non_iid, non_iid_strategy, b, c, e in params_run:
    os.system(order.format(dataset, model, non_iid, non_iid_strategy, b, c, e, file_name,
                           num_clients, max_epochs, lr, optimizer, upload_name_filter,
                           upload_sparse, upload_strategy))
