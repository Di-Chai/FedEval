#!/usr/bin/env bash

# Ministation
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d mnist

# GPU5
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d semantic140
docker run -it --rm --gpus '"device=1"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d femnist

# GPU1
docker run -it --rm --gpus '"device=0"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d celeba
docker run -it --rm --gpus '"device=1"' -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d shakespeare