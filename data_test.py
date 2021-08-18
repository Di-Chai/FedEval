from FedEval.dataset import mnist, cifar10, femnist, celeba


fed_data = femnist(6000, './')

fed_data.non_iid_data(save_file=False)

"""
mnist: 70000
femnist: 3500 Clients
cifar10
"""