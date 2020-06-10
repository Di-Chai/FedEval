# Configs

## Data Config

```yaml
data_dir: data
dataset: mnist
input_shape:
  celeba:
    image:
    - 109
    - 89
    - 3
    label:
    - 2
  cifar10:
    image:
    - 32
    - 32
    - 3
    label:
    - 10
  cifar100:
    image:
    - 32
    - 32
    - 3
    label:
    - 100
  femnist:
    image:
    - 28
    - 28
    - 1
    label:
    - 62
  mnist:
    image:
    - 28
    - 28
    - 1
    label:
    - 10
non-iid: 0 
non-iid-strategy: iid
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
|     data_dir     | The output dir of the clients' data                          |
|     dataset      | Dataset name, mnist / cifar10 / cifar100 / femnist / mnist   |
|   input_shape    | The image shape that will be used by the server to build the model; <br />You need to add inputs_shape for your own dataset here. |
|     non-iid      | 0 for IID setting; <br />>=1 for non-IID setting, <br />When the datasets are mnist, cifar10, cifar100, the non-iid means the # of image classes hold by each client. |
| non-iid-strategy | "natural" strategy for femnist and celebA dataset<br />"average" for mnist, cifar10 and cifar100 |
|    normalize     | Bool. If tue, normalize the image to 0~1                     |
|   sample_size    | Number of Images holf by each client                         |
|   shared_data    | If shared_data > 0, the system will choose the correponding number of image from each client <br />to build shard dataset for all the participants |
|  train_val_test  | Split the data to train, val, and test                       |

## Model Config

```yaml
LeNet:
  activation: relu
  optimizer: adam
  pooling: max
MLP:
  activation: relu
  dropout: 0.2
  optimizer: Adam
  units:
  - 512
  - 512
MobileNet:
  alpha: 0.35
  dense_units:
  - 256
  - 256
  optimizer: adam
  weights: imagenet
Model: LeNet
ResNet50:
  dense_units:
  - 256
  - 256
  optimizer: adam
upload:
  upload_name_filter:
  - None
  upload_sparse: 1.0
  upload_strategy: no-compress
```



Three models are placed inside the system: MLP, LeNet, and MobileNet. You can add your own model and put the config in this file.

|    Config Name     | Description                                                  |
| :----------------: | :----------------------------------------------------------- |
|       Model        | Current using model name                                     |
| upload_name_filter | (List) Use the string in this list to filter all the gradients before uploading, and remove the parameters that the name contains the string in the list.<br />E.g., upload_name_filter=['Adam'] do not upload parameters for Adam optimizer. |
|    input_shape     | The image shape that will be used by the server to build the model; <br />You need to add inputs_shape for your own dataset here. |
|   upload_sparse    | float between 0 and 1, where 1 means no-compression          |
|  upload_strategy   | The strategy in the uploading, current have two choices : "no-compress" and "compress". More features will be added in the future. |

## Runtime Config

```yaml
clients:
  bandwidth: 100Mbit
  local_batch_size: 1000
  local_rounds: 1
  lr: 1e-4
  num_clients: 10
  script: client.py
docker:
  image: fleval:v1
server:
  MAX_NUM_ROUNDS: 5
  MIN_NUM_WORKERS: 10
  NUM_CLIENTS_CONTACTED_PER_ROUND: 10
  NUM_TOLERATE: 20
  ROUNDS_BETWEEN_TEST: 1
  ROUNDS_BETWEEN_VALIDATIONS: 1
  host: server
  listen: server
  port: 8200
  save_gradients: True
  script: server.py
```



The runtime config contains the parameters that will be used in the FL training, and it contains two parts: the client and server.

|           Config Name           | Description                                                  |
| :-----------------------------: | :----------------------------------------------------------- |
|            bandwidth            | Bandwidth for the clients in the uploading and downloading<br />We do not restrict the bandwidth for the server. |
|      local_batch_size (B)       | Local batch size                                             |
|               lr                | learning rate for clients                                    |
|           num_clients           | number of clients                                            |
|             script              | The script for clients. <br />We provided a template "client.py".<br />Actually you do not need to modify this file in most cases. |
|          docker image           | the docker image that will be used for both client and server. |
|         MAX_NUM_ROUNDS          | max number of training rounds                                |
|         MIN_NUM_WORKERS         | the minimum number of clients before start training          |
| NUM_CLIENTS_CONTACTED_PER_ROUND | number of clients that participate in training in each round |
|          NUM_TOLERATE           | early stopping patience                                      |
| ROUNDS_BETWEEN_TEST/VALIDATIONS | number of round between test/validation                      |
|              host               | the IP address that clients connect to the server, could be set to host-name or 'server' (the name of the container network) |
|             listen              | the listen address for the server, could be set to 0.0.0.0 or 'server' (the name of the container network) |
|              port               | port, e.g., 8080                                             |
|         save_gradients          | bool, if set to true, the server will save the parameters in the training |
|             script              | The script for the server. <br />We provided a template "server.py".<br />Actually you do not need to modify this file in most cases. |