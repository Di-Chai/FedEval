## Reproduce the results in the paper

#### Reproducing scripts

We provide the following scripts for reproducing the results in our paper

|                     Experiments                      |                     Reproducing Scripts                      |
| :--------------------------------------------------: | :----------------------------------------------------------: |
|         Grid search of B, C, and E in FedAvg         |                     Dataset_Tune_BCE.py                      |
|       Tuning the optimizers and learning rates       |                    Dataset_Tune_Op_LR.py                     |
| Accuracy, Communication, Time consumption Evaluation |                       Dataset_Eval.py                        |
|                  Privacy Evaluation                  | DLG_Attack_MultiThread.py<br />DLG_Attack_Metric.py<br />FC_Attack_MultiThread.py<br />FC_Attack_Metric.py |
|                Robustness Evaluation                 |                      Dataset_Non_IID.py                      |

where the Dataset can be MNIST, FEMNIST, and CelebA, e.g., MNIST_Tune_BCE.py.

#### Running the scripts

```bash
cd PaperScripts
```

Grid search of B, C, and E in FedAvg

```bash
# MNIST Dataset, output file: MNIST_Tune_BCE_trials.txt
python MNIST_Tune_BCE.py

# FEMNIST Dataset, output file: FEMNIST_Tune_BCE_trials.txt
python FEMNIST_Tune_BCE.py

# CelebA Dataset, output file: CelebA_Tune_BCE_trials.txt
python CelebA_Tune_BCE.py
```

Tuning the optimizers and learning rates

```bash
# MNIST Dataset, output file: MNIST_Tune_Op_LR_trials.txt
python MNIST_Tune_Op_LR.py

# FEMNIST Dataset, output file: FEMNIST_Tune_Op_LR_trials.txt
python FEMNIST_Tune_Op_LR.py

# CelebA Dataset, output file: CelebA_Tune_Op_LR_trials.txt
python CelebA_Tune_Op_LR.py
```

Accuracy, Communication, and Time consumption evaluation.

```bash
# MNIST Dataset, output file: MNIST_Eval_trials.txt
python MNIST_Eval.py

# FEMNIST Dataset, output file: FEMNIST_Eval_trials.txt
python FEMNIST_Eval.py

# CelebA Dataset, output file: CelebA_Eval_trials.txt
python CelebA_Eval.py
```

Privacy experiments

```bash
# 1.1 Start the DLG attack using multi-threads
python DLG_Attack_MultiThread.py

# 1.2 Calculate the label accuracy and L2 distance using the results from 1.1
python DLG_Attack_Metric.py

# 2.1 Start the FC attack using multi-threads
python FC_Attack_MultiThread.py

# 2.2 Calculate the label accuracy and L2 distance using the results from 2.1
python FC_Attack_Metric.py
```

Robustness evaluation

```bash
# MNIST Dataset, output file: MNIST_Non_IID_trials.txt
python MNIST_Non_IID.py

# FEMNIST Dataset, output file: FEMNIST_Non_IID_trials.txt
python FEMNIST_Non_IID.py

# CelebA Dataset, output file: CelebA_Non_IID_trials.txt
python CelebA_Non_IID.py
```

