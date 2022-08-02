#!/usr/bin/env bash

cd ..

export FED_EVAL_REPEAT=10
export FED_EVAL_LOG_DIR=log/JMLRSparsity

# FedSTC - Non-IID
sudo $(which python) trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t sparsity -n 1 -e run -d mnist
