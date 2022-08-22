#!/usr/bin/env bash

cd ..

export FED_EVAL_REPEAT=1
export FED_EVAL_LOG_DIR=log/JMLR

# TuneLR for FedSGD
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d femnist  # GPU2
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d celeba  # GPU5
docker run -it --rm --gpus '"device=6"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d sentiment140  # GPU6
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d shakespeare  # GPU1
docker run -it --rm --gpus '"device=4"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -n 1 -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d mnist  # GPU6

# TuneLR for Central
docker run -it --rm --gpus '"device=1"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d femnist # GPU2
docker run -it --rm --gpus '"device=1"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d celeba # GPU5
docker run -it --rm --gpus '"device=7"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d sentiment140 # GPU6
docker run -it --rm --gpus '"device=1"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d shakespeare # GPU1
docker run -it --rm --gpus '"device=5"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s LocalCentral -c configs/quickstart -m local -i true -n 1 -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d mnist # GPU6

# TuneLR for Local
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d femnist  # WorkStation
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d celeba  # WorkStation
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d sentiment140  # WorkStation
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d shakespeare  # WorkStation
python trial.py -s LocalCentral -c configs/quickstart -m local -i true -n 1 -t lr -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d mnist  # WorkStation (Done)

# TuneLR for FedSTC
export FED_EVAL_REPEAT=1
export FED_EVAL_LOG_DIR=log/JMLRSTC
export CUDA_VISIBLE_DEVICES=0,1,2,3
python trial_by_cl.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m without-docker -i true -t lr -e run -d femnist
python trial_by_cl.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m without-docker -i true -t lr -e run -d celeba
python trial_by_cl.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m without-docker -i true -t lr -e run -d sentiment140
python trial_by_cl.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m without-docker -i true -t lr -e run -d shakespeare
python trial_by_cl.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m without-docker -i true -n 1 -t lr -e run -d mnist
# Or
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d femnist
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d celeba
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d sentiment140
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d shakespeare
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -t lr -e run -d mnist

# TuneLR for FedAvg
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d femnist  # WorkStation
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d celeba   # GPU1
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d sentiment140  # GPU2
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d shakespeare
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -t lr -e run -d mnist # Done

# TuneLR for FedOpt
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d femnist
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d celeba
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d sentiment140
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d shakespeare
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -t lr -e run -d mnist

# TuneLR for FedProx
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d femnist
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d celeba
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d sentiment140
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -t lr -e run -d shakespeare
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -t lr -e run -d mnist

python trial_by_cl.py -s FedProx -r 1 -l log\DEBUG -c configs\quickstart -m local -i true -t lr -e run -d shakespeare