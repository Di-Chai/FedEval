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

Strategy-specific configuraitons:

|Federated Strategy|Config Name|Description|
|:--:|:--:| :--|
|FedSTC|sparsity| |
|FedProx|mu| the /mu parameter in FedProx, a scaler that measures the approximation between the local model and the global model.|
|FedOpt|tau| TBD|
|FedOpt|beta1| TBD|
|FedOpt|beta2| TBD|
|FedOpt|opt_name| TBD|
|FedSCA/FedOpt|eta|the learning rate on the server side.|
|FetchSGD|num_col| the number of columns in FetchSGD. TODO(fgh) more specific|
|FetchSGD|num_row|the number of rows in FetchSGD.|
|FetchSGD|num_block| the number of blocks in FetchSGD.|
|FetchSGD|top_k| the number of top items during the TopK unsketching.|

### MLModel

All configuraitons of the machine learning model is available in [TensorFlow Core v2 APIs](https://tensorflow.google.cn/api_docs/python/tf), for the code in the model module is conducted with tensorflow-v2. 

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


## Access Configuraitons Anywhere & Anytime

Once the `ConfigurationManager` is directly constructed with `RawConfigurationDict`s or deserialized from some medium representations (e.g., JSON and YAML within string or file system), it can not be modified any more. While there are some exceptions:

1. the filename of the configurations: these are used to (de)serialize the configuraitons during runtime.
2. the role of an instance: the role of an instance should be set as soon as the role of the instance is clear for once and only once. Currently, the role is set during the construction of Server/Client in the `role` module.
3. encoding: the encoding for the (de)serialization operations.

It is highly recommend to have `ConfigurationManager` constructed(or, initialized) once all the modifications on the  `RawConfigurationDict` were done. You can initialize it as follows:

```Python
from ..config.configuration import ConfigurationManager
# 1. constructed from raw config dicts
# ... read configuration dicts ...
ConfigurationManager(data_config, model_config, runtime_config)

# 2. deserialized from file (for example here, from_yamls and from_jsons are also available.)
cfg_mgr = ConfigurationManager.from_files(data_cfg_path, model_cfg_path, runtime_cfg_path)
```

Currently, the initialization of the `ConfigurationManager` is conducted in run.py and run_util.py.

After the initialization, you can access all the configurations from anywhere and at anytime.

```Python
from ..config.configuration import ConfigurationManager

cfg_mgr = ConfigurationManager()
d_cfg, mdl_cfg, rt_cfg = cfg_mgr.data_config, cfg_mgr.model_config, cfg_mgr.runtime_config
```

Noticed that all the configigurations have been reorganized as objects, thus you can access any items in the configurations just like accessing attributes of an object, with concrete type hints and type conversion:

```Python
d_name: str = d_cfg.dataset_name
```

You cannot set the attributes with a new value, for the setters of these properties are not implemented for security considerations. You will get an `AttributeError` if you assigned it with a new value.

## Want to Modify the Configuration Scheme?

Go ahead into `config` module. More specifically, into 'config/configuration.py', where all the preset configuration schemes are defined. At least two steps are required to add a new configuration item:

1. define the key name in the dict;
2. (optional but recommended) give a default value of this new item;
3. add a new property in the corresponding `_Config` class.

Now, we take 'dataset' in '1_data_config.yml' as an example for illustration. First, let's define its key name in the dict.

```Python
# others are omitted
_D_NAME_KEY = 'dataset' # here!
_DEFAULT_D_CFG: RawConfigurationDict = {
    _D_NAME_KEY: 'mnist', # and give it a default value
    # others are omitted
}
```

Then, add a property in `_DataConfig`.

```Python
class _DataConfig(_Configuraiton):
    @property
    def dataset_name(self) -> str:
        return self._inner[_D_NAME_KEY]
```

Now, you can configure the name of dataset in the configuration files and access it in the code.
