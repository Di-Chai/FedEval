from FedEval.dataset import mnist, cifar10, femnist, celeba


fed_data = mnist(100, 'data_test')

fed_data.non_iid_data(save_file=True, non_iid_class=1, strategy='average')

"""
mnist: 70000
femnist: 3500 Clients
cifar10
"""