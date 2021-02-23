import os
from utils import grid_search
os.chdir('../')


# the number of repeated experiments
repeat = 1
max_epoch = 5

"""
1 MNIST Dataset
"""

# 1.1 FedSGD grid-search on learning rates
grid_search(
    execution='run', dataset='mnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='MLP', fed_model='FedSGD', output='MNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[1000], tune_C=[1.0], tune_E=[1], tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)
grid_search(
    execution='run', dataset='mnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='LeNet', fed_model='FedSGD', output='MNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[1000], tune_C=[1.0], tune_E=[1], tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)
# 1.2 FedAvg grid-search on B, C, E
grid_search(
    execution='run', dataset='mnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='MLP', fed_model='FedAvg', output='MNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[32, 16, 8, 4, 2, 1], tune_C=[0.1], tune_E=[2, 4, 8, 16, 32, 64],
    tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)
grid_search(
    execution='run', dataset='mnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='LeNet', fed_model='FedAvg', output='MNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[32, 16, 8, 4, 2, 1], tune_C=[0.1], tune_E=[2, 4, 8, 16, 32, 64],
    tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)

"""
2 FEMNIST
"""

# 1.1 FedSGD grid-search on learning rates
grid_search(
    execution='run', dataset='femnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='MLP', fed_model='FedSGD', output='FEMNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[1000], tune_C=[1.0], tune_E=[1], tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)
grid_search(
    execution='run', dataset='femnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='LeNet', fed_model='FedSGD', output='FEMNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[1000], tune_C=[1.0], tune_E=[1], tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)
# 1.2 FedAvg grid-search on B, C, E
grid_search(
    execution='run', dataset='femnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='MLP', fed_model='FedAvg', output='FEMNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[32, 16, 8, 4, 2, 1], tune_C=[0.1], tune_E=[2, 4, 8, 16, 32, 64],
    tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)
grid_search(
    execution='run', dataset='femnist', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='LeNet', fed_model='FedAvg', output='FEMNIST_ParamSearch.csv',
    repeat=repeat, tune_B=[32, 16, 8, 4, 2, 1], tune_C=[0.1], tune_E=[2, 4, 8, 16, 32, 64],
    tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)

"""
3 CelebA
"""

# 1.1 FedSGD grid-search on learning rates
grid_search(
    execution='run', dataset='celeba', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='LeNet', fed_model='FedSGD', output='CelebA_ParamSearch.csv',
    repeat=repeat, tune_B=[1000], tune_C=[1.0], tune_E=[1], tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)
# 1.2 FedAvg grid-search on B, C, E
grid_search(
    execution='run', dataset='celeba', mode='local', config='configs/local',
    optimizer='adam', upload_optimizer='True', max_epoch=max_epoch,
    ml_model='LeNet', fed_model='FedAvg', output='CelebA_ParamSearch.csv',
    repeat=repeat, tune_B=[32, 16, 8, 4, 2, 1], tune_C=[0.1], tune_E=[2, 4, 8, 16, 32, 64],
    tune_LR=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
)