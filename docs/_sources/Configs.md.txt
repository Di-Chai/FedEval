# Configs

Following we give detailed comments for each parameter used in the config files.

## Data Config

```yaml
# data_dir: The output dir of the clients' data
data_dir: data
# dataset: Dataset name, mnist / femnist / celeba / semantic140
dataset: semantic140
# non-iid: True or False, controling the non-IID setting
non-iid: True
# non-iid-class: The # of image classes hold by each client, used in mnist, cifar10, cifar100, only works when non-iid=True
non-iid-class: 1
# non-iid-strategy: which strategy will be used in the non-IID setting
# E.g., "natural" strategy for femnist, celebA, semantic140 dataset, "average" for mnist, cifar10 and cifar100
non-iid-strategy: average # Only work when non-iid = True
# normalize: True or False, controling whether the data will be normalized
normalize: true
# sample_size: Each client's sample size
sample_size: 300
# train_val_test: Split the data to train, val, and test
train_val_test:
  - 0.8
  - 0.1
  - 0.1
```

## Model Config

```yaml
FedModel:
	# which FL mechanism will be used, could be: FedSGD, FedAvg, FedOpt, FedProx, FedSTC, FedSCA
  name: FedAvg
  # B: client's local batch size (Shared By all mechanisms)
  B: 32
  # C: percent of seleced clients during training (Shared By all mechanisms)
  C: 0.1
  # E: clients' local training passes (Shared By all mechanisms)
  E: 1
  # max_rounds: Maximum number of training rounds (Shared By all mechanisms)
  max_rounds: 10000
  # num_tolerance: Used for early-stopping (Shared By all mechanisms)
  # E.g., when num_tolerance=300, the training stops if the validation loss is not getting lower for 300 rounds
  num_tolerance: 300
  # rounds_between_val: (Shared By all mechanisms)
  # Frequency of doing evaluation, default to evaluate after each round of training 
  rounds_between_val: 1
  # sparsity: Compression retio used by FedSTC
  sparsity: 0.01
  # mu: regularization parameter used by FedProx
  mu: 0.01
  # parameters used by FedOpt
  tau: 0.0001
  beta1: 0.9
  beta2: 0.99
  opt_name: 'fedyogi'
  # Server-side learning rate, used by FedOpt and FedSCA
  eta: 1.0
MLModel:
	# which ML model will be used, could be: MLP, LeNet, StackedLSTM
  name: MLP
  # activation method (Shared By all Method)
  activation: relu
  # Dropouts used in MLP
  dropout: 0.2
  # Hidden layer units used in MLP
  units:
    - 512
    - 512
  # Optimizer parameters (Shared By all Method)
  optimizer:
    name: sgd
    lr: 0.1
    momentum: 0
  # Type of loss function (Shared By all Method)
  loss: categorical_crossentropy
 	# Additional metrics apart from the loss metric (Shared By all Method)
  metrics:
    - accuracy
```

## Runtime Config

```yaml
clients:
	# bandwidth: Networking bandwidth between the clients and server
  bandwidth: 100Mbit
docker:
	# image: docker image ID
  image: fedeval:v1
  # num_containers: number of containers
  num_containers: 40
  # enable_gpu: true or false
  enable_gpu: True
  # num_gpu: how many gpus do you want to use
  # the system will by-default pick gpus from the first one 
  num_gpu: 8
server:
	# host: address clients will connect to; Here the "server" is the name of container-network
	# if you are running with more than one mechine, change "host" to the server's ip address
  host: server
  # listen: address server will listen on; Here the "server" is the name of container-network
  # "listen" is the same with "host" when running experiments locally
  # if you are running with more than one mechine, change "listen" to 0.0.0.0
  listen: server
  # num_clients: the number of clients
  num_clients: 100
  # post: port server will listen on 
  port: 8000
# Config the following parameters, if you are using more than one mechine in one experiment
machines:
	# server config (doing the aggregation)
  server:
    host: 10.173.1.22
    port: 22
    user_name: ubuntu
    # The remote dir that stored the project file, e.g., /home/user_mame/FedEval
    dir: /home/chaidi/FedEval
    # Path to the secret key, used to login the remote server
    key: id_rsa
  # machine config (doing the training)
  m1:
  	# the machine's ip could be that same with the server
  	# i.e., you can chose one worker as the central server which performs the aggregation
    host: 10.173.1.22
    port: 22
    # capacity: how many containers can this machine hold
    capacity: 100
    user_name: ubuntu
    dir: /home/chaidi/FedEval
    key: id_rsa
# log dir name
log_dir: log/debug
```

