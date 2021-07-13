#!/usr/bin/env bash

# MNIST IID
# python3 -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i false -r 5 -e run
# python3 -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i false -r 10 -e run

# MNIST Non-IID
# python3 -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 1 -r 10 -e run
# python3 -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 2 -r 10 -e run
# python3 -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 3 -r 10 -e run
# python3 -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 1 -r 10 -e run
# python3 -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 2 -r 10 -e run
# python3 -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 3 -r 10 -e run

# FEMNIST IID
python3 -W ignore trial.py -d femnist -s FedSGD -c configs/workstation -m server -i false -r 10 -e run
python3 -W ignore trial.py -d femnist -s FedAvg -c configs/workstation -m server -i false -r 10 -e run

# FEMNIST Non-IID
python3 -W ignore trial.py -d femnist -s FedSGD -c configs/workstation -m server -i true -r 10 -e run
python3 -W ignore trial.py -d femnist -s FedAvg -c configs/workstation -m server -i true -r 10 -e run

# CelebA IID
python3 -W ignore trial.py -d celeba -s FedSGD -c configs/workstation -m server -i false -r 10 -e run
python3 -W ignore trial.py -d celeba -s FedAvg -c configs/workstation -m server -i false -r 10 -e run

# CelebA Non-IID
python3 -W ignore trial.py -d celeba -s FedSGD -c configs/workstation -m server -i true -r 10 -e run
python3 -W ignore trial.py -d celeba -s FedAvg -c configs/workstation -m server -i true -r 10 -e run

# Sent140 IID
# python3 -W ignore trial.py -d semantic140 -s FedSGD -c configs/workstation -m server -i false -r 8 -e run
# python3 -W ignore trial.py -d semantic140 -s FedAvg -c configs/workstation -m server -i false -r 6 -e run

# Sent140 Non-IID
# python3 -W ignore trial.py -d semantic140 -s FedSGD -c configs/workstation -m server -i true -r 10 -e run
# python3 -W ignore trial.py -d semantic140 -s FedAvg -c configs/workstation -m server -i true -r 10 -e run

