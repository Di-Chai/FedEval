#!/usr/bin/env bash

cd ..

export FED_EVAL_REPEAT=10
export FED_EVAL_LOG_DIR=log/JMLRSummary

# FedSTC - IID
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i false -n 1 -e run -d mnist
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i false -e run -d femnist
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i false -e run -d celeba
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i false -e run -d semantic140
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i false -e run -d shakespeare
