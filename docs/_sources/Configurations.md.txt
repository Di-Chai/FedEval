# Get to know the three configurations

## Data Config

```yaml
data_dir: data
dataset: mnist
non-iid: 0
non-iid-class: 1 # Only work when non-iid = True
non-iid-strategy: average
normalize: true
sample_size: 300
shared_data: 0
train_val_test:
- 0.8
- 0.1
- 0.1
```

|   Config Name    | Description                                                  |
| :--------------: | :----------------------------------------------------------- |
|     data_dir     | The output directory of the clients' data                    |
|     dataset      | Dataset name, mnist / cifar10 / cifar100 / femnist / mnist   |
|     non-iid      | Bool. |
|non-iid-class|The number of image classes hold by each client when non-iid is True|
| non-iid-strategy | "natural" strategy for femnist and celebA dataset<br />"average" for mnist, cifar10 and cifar100 |
|    normalize     | Bool. If true, normalize the image to 0~1                    |
|   sample_size    | Number of Images hold by each client                        |
|   shared_data    | If shared_data > 0, the system will choose the corresponding number of image from each client to build shard dataset for all the participants |
|  train_val_test  | Split the data to train, validation, and test. This list indicates the shares of each one |

## Model Config

```yaml
FedModel:
  name: FedSGD
  # Shared params
  B: 1000
  C: 1.0
  E: 1
  max_rounds: 1000
  num_tolerance: 100
  rounds_between_val: 1
  # FedSTC
  sparsity: 0.01
  # FedProx
  mu: 0.01
  # FedOpt
  tau: 0.0001
  beta1: 0.9
  beta2: 0.99
  opt_name: 'fedyogi'
  # Server LR, used by FedOpt and FedSCA
  eta: 1.0
MLModel:
  name: MLP
  activation: relu
  dropout: 0.2
  units:
    - 512
    - 512
  optimizer:
    name: sgd
    lr: 0.1
    momentum: 0
  loss: categorical_crossentropy
  metrics:
    - accuracy
```

Three models are placed inside the system: MLP, LeNet, and MobileNet. You can add your own model and put the config in this file.

### FedModel

|             Config Name              | Description                                                  |
| :----------------------------------: | :----------------------------------------------------------- |
|                 name                 | The name of federated strategy                               |
|            num_tolerance             | Early stopping patience                                      |
|              max_rounds              | The maximum rounds that can be reached                       |
|          rounds_between_val          | The number of round between test or validation               |
|                  B                   | The local minibatch size used for the client updates         |
|                  C                   | The fraction of clients that perform computation on each round |
|                  E                   | The number of training passes each client makes over its local dataset on each round |
| To be done: strategy specific params |                                                              |

### MLModel

To be done.

## Runtime Config

```yaml
clients:
  bandwidth: 100Mbit
docker:
  image: fedeval:v4
server:
  host: server
  listen: server
  num_clients: 10
  port: 8080
log_dir: log/quickstart
```

The runtime config contains the parameters that will be used in the FL training, and it contains two parts: the client and server.

| Config Name  | Description                                                  |
| :----------: | :----------------------------------------------------------- |
|  bandwidth   | Bandwidth for the clients in the uploading and downloading<br />We do not restrict the bandwidth for the server. |
| num_clients  | The number of clients                                        |
| docker image | The docker image that will be used for both client and server. |
|     host     | The IP address that clients connect to the server, could be set to host-name or 'server' (the name of the container network) |
|    listen    | The listen address for the server, could be set to 0.0.0.0 or 'server' (the name of the container network) |
|     port     | Port, e.g., 8080                                             |
|   log_dir    | Path for saving the log and results                          |

