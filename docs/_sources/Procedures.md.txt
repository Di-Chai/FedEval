### Running Multiple Experiments (using scripts)

#### Start the experiments

In the last section (QuickStart), we presented how to start a single experiment at the terminal. Although we can start different trials by modifying the parameters in the config file, the terminal way of starting the experiments is not convenient. 

Thus, we introduce a new way to run multiple experiments using the scripts.

Briefly, we use the `FedEval.run_util.run` function, which is a higher-level scheduler based on `FedEval.run`, ssh and scp. Specifically, it direct the whole lifecycle of each instances in an experiment, including:

1. compose and dispatch the dockerfile and configuration files;
1. prepare the dataset for each client;
1. launch the experiments;
1. stop the experiment at any time you want.

Here's an example which aimed at conducting a grid search for learning rate `lr`.

```python
from FedEval.run_util import run

params = {
    'data_config': {
        'dataset': 'mnist',
        'non-iid': False,
        'sample_size': 300,
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
    # update learning rate in configuration
    params['model_config']['MLModel']['optimizer']['lr'] = lr 
    run(exec='run', mode='local', config='configs/quickstart', new_config=config + '_tmp', **params)
```

Noted that the `params` passed into `run`  will override the configurations specified in `config='configs/quickstart'`. And the new configurations will be saved to `new_config=config + '_tmp'`, i.e., keeping the original file untouched.

After run this script in terminal,  you can visit `http://127.0.0.1:8080/dashboard` have an overview of the experiments' status.

#### Check the results

Logs, results and model weight records are stored under `log` directory, classified by their names and time when they run the experiments.

Using the following commend to collect the results from the logs:

```bash
sudo docker run -it --rm -v $(pwd):$(pwd) -w $(pwd) fedeval:v1 python -W ignore -m FedEval.run_util -e log -p log/quickstart/Server
```

