#!/usr/bin/env bash

export FED_EVAL_REPEAT=10
export FED_EVAL_LOG_DIR=log/tune_lr_jmlr

# TuneLR for FedSGD
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d femnist
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d celeba
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d semantic140
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d shakespeare
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -n 1 -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d mnist

# TuneLR for Central
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d femnist
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d celeba
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d semantic140
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d shakespeare
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -n 1 -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d mnist

# TuneLR for Local
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d femnist
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d celeba
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d semantic140
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d shakespeare
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -n 1 -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d mnist

# TuneLR for FedAvg
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d femnist
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d celeba
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d semantic140
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d shakespeare
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -t lr -e run -d mnist

# TuneLR for FedOpt
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d femnist
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d celeba
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d semantic140
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d shakespeare
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -t lr -e run -d mnist

# TuneLR for FedProx
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d femnist
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d celeba
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d semantic140
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d shakespeare
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -t lr -e run -d mnist
