## Reproduce the Benchmark Results

#### Initialize the Environments

Please follow the instructions from [previous sections](Environment.md) to prepare the environments.

To check whether the environment is parperly settled, you can run the test examples in [quickstart](QuickStart.md) and [multiple experiments using scripts](Procedures.md).

#### Reproduce the Benchmark Results

We have fixed the finetuned parameters in `trial.py`, and all the benchmark results could be repriduced using this script. 

```bash
# -d dataset
# -s FL strategy
# -c configs
# -m mode, local or server
# -i iid, ture or false
# -r repeat times (e.g.,, repeat the trial for 10 times using the same parameter)
# -e run or stop
python3 -W ignore trial.py -d mnist -s FedSGD -c configs/local -m local -i false -r 10 -e run
```

Since there are many experiments, to simplify the reproduction, we have summarized the experiments in `reproduce.py` and `reproduce.sh`.

The benchmark results could be reproduced by:

```bash
python reproduce.py
```

or 

```bash
sh reproduce.sh
```