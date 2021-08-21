import os

# MNIST IID
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i false -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i false -r 10 -e run")

# MNIST Non-IID
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 1 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 2 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 3 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 1 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 2 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 3 -r 10 -e run")

# FEMNIST IID
# os.system("python -W ignore trial.py -d femnist -s FedSGD -c configs/workstation -m server -i false -r 10 -e run")
# os.system("python -W ignore trial.py -d femnist -s FedAvg -c configs/workstation -m server -i false -r 10 -e run")

# FEMNIST Non-IID
# os.system("python -W ignore trial.py -d femnist -s FedSGD -c configs/workstation -m server -i true -r 10 -e run")
# os.system("python -W ignore trial.py -d femnist -s FedAvg -c configs/workstation -m server -i true -r 10 -e run")

# CelebA IID
# os.system("python -W ignore trial.py -d celeba -s FedSGD -c configs/workstation -m server -i false -r 10 -e run")
# os.system("python -W ignore trial.py -d celeba -s FedAvg -c configs/workstation -m server -i false -r 10 -e run")

# CelebA Non-IID
# os.system("python -W ignore trial.py -d celeba -s FedSGD -c configs/workstation -m server -i true -r 10 -e run")
# os.system("python -W ignore trial.py -d celeba -s FedAvg -c configs/workstation -m server -i true -r 10 -e run")

# Sent140 IID
# os.system("python -W ignore trial.py -d semantic140 -s FedSGD -c configs/workstation -m server -i false -r 10 -e run")
# os.system("python -W ignore trial.py -d semantic140 -s FedAvg -c configs/workstation -m server -i false -r 10 -e run")

# Sent140 Non-IID
# os.system("python -W ignore trial.py -d semantic140 -s FedSGD -c configs/workstation -m server -i true -r 10 -e run")
# os.system("python -W ignore trial.py -d semantic140 -s FedAvg -c configs/workstation -m server -i true -r 10 -e run")


# FedProx on different datasets
# os.system("python -W ignore trial.py -d mnist -s FedProx -c configs/workstation -m server -i true -n 1 -r 5 -e run")
os.system("python -W ignore trial.py -d femnist -s FedProx -c configs/workstation -m server -i true -r 5 -e run")
os.system("python -W ignore trial.py -d celeba -s FedProx -c configs/workstation -m server -i true -r 5 -e run")
os.system("python -W ignore trial.py -d semantic140 -s FedProx -c configs/workstation -m server -i true -r 5 -e run")
