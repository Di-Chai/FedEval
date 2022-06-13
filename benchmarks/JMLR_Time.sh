#!/usr/bin/env bash

cd ..

export FED_EVAL_REPEAT=1
export FED_EVAL_LOG_DIR=log/JMLRTime

export FED_EVAL_PYTHON=$(which python)

# FedSGD - Non-IID
sudo $FED_EVAL_PYTHON trial.py -s FedSGD -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
sudo $FED_EVAL_PYTHON trial.py -s FedSGD -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
sudo $FED_EVAL_PYTHON trial.py -s FedSGD -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
sudo $FED_EVAL_PYTHON trial.py -s FedSGD -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
sudo $FED_EVAL_PYTHON trial.py -s FedSGD -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist

# FedSTC - Non-IID
sudo $FED_EVAL_PYTHON trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
sudo $FED_EVAL_PYTHON trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
sudo $FED_EVAL_PYTHON trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
sudo $FED_EVAL_PYTHON trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
sudo $FED_EVAL_PYTHON trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist

# FedAvg - Non-IID
sudo $FED_EVAL_PYTHON trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
sudo $FED_EVAL_PYTHON trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
sudo $FED_EVAL_PYTHON trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
sudo $FED_EVAL_PYTHON trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
sudo $FED_EVAL_PYTHON trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist

# FedOpt - Non-IID
sudo $FED_EVAL_PYTHON trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
sudo $FED_EVAL_PYTHON trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
sudo $FED_EVAL_PYTHON trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
sudo $FED_EVAL_PYTHON trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
sudo $FED_EVAL_PYTHON trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist

# FedProx - Non-IID
sudo $FED_EVAL_PYTHON trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
sudo $FED_EVAL_PYTHON trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
sudo $FED_EVAL_PYTHON trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
sudo $FED_EVAL_PYTHON trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
sudo $FED_EVAL_PYTHON trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist
