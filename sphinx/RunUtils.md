## Get to know the FedEval.run and FedEval.run_util

#### FedEval.run.\_\_main\_\_

This function is a CLI interface, which is used to:

* prepare dataset according to 1_data_config.yml;

* compose dockerfile for both local and server(remote) mode;
* launch a single experiment;

`FedEval.run.__main__`  is suitable for *small scale* experiments, such as prototype iteration, debug. You can directly use them in containers like this:

![docker run command](images/docker_run_params.png)

where the dark part issues a new container using the image we build (tagged as fedeval:v4); the blue part are parameters of  `docker run` command, which maps the current directory to the same place in the container and sets working directory as this one; the red part is what we run in the container; and the orange part are parameters of `FedEval.run.__main__`.

Here's an example to launch a local experiment with the preset configurations in `FedEval/configs/quickstart`.

1. Preprocess the dataset: 

   `sudo docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) fedeval:v4 python -m FedEval.run -c configs/quickstart -f data`

1. Generate the local compose-file:

   `sudo docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) fedeval:v4 python -m FedEval.run -c configs/quickstart -f compose-local`

1. Start the experiment as a daemon (remove `-d` if you want to see logs from terminal):

   `sudo docker-compose up -d`

1. After a few seconds, visit `http://127.0.0.1:8080/dashboard` on your browser to have an overview of the experiment's status. 

For full details about this function, please refer to the [documentation]().

#### FedEval.run_util.run

This function is a higher-level scheduler based on `FedEval.run`, ssh and scp. Specifically, it direct the whole lifecycle of each instances in an experiment, including:

1. compose and dispatch dockerfile and configuration files;
1. prepare the dataset for each client;
1. launch train-val-test iterations;
1. stop the experiment at any time you want;

As it is exposed as a function interface, it can be coded in order to handle some large-scale operations, such as grid search, tuning, etc.

Here's an example which aimed at conducting a grid search for learning rate `lr`.

```python
from FedEval.run_util import run

params = {
    'data_config': {
        'dataset': 'mnist',
        'non-iid': True,
        'sample_size': 300,
        'non-iid-class': 2,
        'non-iid-strategy': 'average'
    },
    'model_config': {
        'MLModel': {
            'name': 'MLP',
            'optimizer': {
                'name': 'sgd', 'lr': 0.5, 'momentum': 0
            }
        },
        'FedModel': {
            'name': 'FedAvg', 'B': 16, 'C': 0.1, 'E': 10,
            'max_rounds': 3000, 'num_tolerance': 100,
        }
    },
    'runtime_config': {
        'server': {
            'num_clients': 10
        }
    }
}

for lr in [0.001, 0.01, 0.1, 1]:
    params['model_config']['MLModel']['optimizer']['lr'] = lr # update learning rate in configuration
    run(exec='run', mode='local', config='configs/quickstart', new_config=config + '_tmp', **params)
```

Noted that the `params` passed into `run` in line 32 will override the configurations specified in `config='configs/quickstart'`. In another word, this interface will take in two configurations: One is saved in external YAML files as a default choice; the other one is an optional parameter of function `FedEval.run_util.run`. And the latter one has a higher priority.

After run this script in terminal,  you can visit `http://127.0.0.1:8080/dashboard` have an overview of the experiments' status.

