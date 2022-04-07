#!/usr/bin/env bash

docker run -it --rm --gpus all -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d femnist
docker run -it --rm --gpus all -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d celeba
docker run -it --rm --gpus all -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d semantic140
docker run -it --rm --gpus all -v $(pwd):/fml -w /fml fedeval:v1 python trial.py -s FedSGD -c configs/quickstart -m local -i false -t lr -r 10 -e simulate_fedsgd -d shakespeare