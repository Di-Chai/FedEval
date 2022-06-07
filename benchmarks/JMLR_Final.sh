#!/usr/bin/env bash

export FED_EVAL_REPEAT=10
export FED_EVAL_LOG_DIR=log/JMLRSummary

# FedSGD - Non-IID
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d femnist
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d celeba
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d semantic140
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d shakespeare
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i true -n 1 -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d mnist

# FedSGD - IID
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d femnist
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d celeba
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d semantic140
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d shakespeare
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_fedsgd -d mnist

# Central - Non-IID
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Central -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d femnist
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Central -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d celeba
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Central -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d semantic140
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Central -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d shakespeare
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Central -c configs/quickstart -m local -i true -n 1 -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_central -d mnist

# Local - Non-IID
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Local -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d femnist
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Local -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d celeba
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Local -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d semantic140
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Local -c configs/quickstart -m local -i true -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d shakespeare
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s Local -c configs/quickstart -m local -i true -n 1 -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -e simulate_local -d mnist

# FedSTC - Non-IID
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
python trial.py -s FedSTC -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist

# FedAvg
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
python trial.py -s FedAvg -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist

# FedOpt
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
python trial.py -s FedOpt -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist

# FedProx
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d femnist
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d celeba
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d semantic140
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -e run -d shakespeare
python trial.py -s FedProx -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -i true -n 1 -e run -d mnist
