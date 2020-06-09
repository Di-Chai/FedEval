import os

os.chdir('../')

params_run = [

    # dataset, model, optimizer, upload_name_filter, lr, non_iid, non_iid_strategy, b, c, e

    # MLP FedSGD, best lr 1e-2
    ['mnist', 'MLP', 'adam', 'None', 1e-2, 0, 'iid', 1000, 1.0, 1],

    # MLP FedAvg, best lr 5e-4
    ['mnist', 'MLP', 'adam', 'None', 5e-4, 0, 'iid', 8, 0.1, 16],

    # LeNet FedSGD, best lr 1e-2
    # ['mnist', 'LeNet', 'adam', 'None', 1e-2, 0, 'iid', 1000, 1.0, 1],
    #
    # # LeNet FedAvg, best lr 5e-3
    # ['mnist', 'LeNet', 'adam', 'None', 5e-3, 0, 'iid', 8, 0.1, 16],

]

repeat = 10

num_clients = 100
max_epochs = 2000

upload_sparse = 1.0
upload_strategy = 'no-compress'

order = 'python run_base_local.py --dataset {} --model {} --non-iid {} --non-iid-strategy {} ' \
        '--B {} --C {} --E {} --file_name {} --num_clients {} --max_epochs {} --lr {} --optimizer {} ' \
        '--upload_name_filter {} --upload_sparse {} --upload_strategy {} ' \
        '--sudo no'

file_name = os.path.basename(__file__).strip('run_ .py') + '_trials.txt'

for r in range(repeat):
    for dataset, model, optimizer, upload_name_filter, lr, non_iid, non_iid_strategy, b, c, e in params_run:
        os.system(order.format(dataset, model, non_iid, non_iid_strategy,
                               b, c, e, file_name,
                               num_clients, max_epochs, lr, optimizer, upload_name_filter,
                               upload_sparse, upload_strategy))
