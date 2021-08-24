import os

# MNIST IID
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i false -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i false -r 10 -e run")

# MNIST Non-IID
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 1 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 1 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedSGD -c configs/workstation -m server -i true -n 3 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 1 -r 10 -e run")
# os.system("python -W ignore trial.py -d mnist -s FedAvg -c configs/workstation -m server -i true -n 1 -r 10 -e run")
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


# FedSTC
os.system("python -W ignore trial.py -d mnist -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d mnist -s FedSTC -c configs/workstation -m server -i true -n 1 -r 1 -e run")
os.system("python -W ignore trial.py -d femnist -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d femnist -s FedSTC -c configs/workstation -m server -i true -r 1 -e run")
os.system("python -W ignore trial.py -d celeba -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d celeba -s FedSTC -c configs/workstation -m server -i true -r 1 -e run")
os.system("python -W ignore trial.py -d semantic140 -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d semantic140 -s FedSTC -c configs/workstation -m server -i true -r 1 -e run")

os.system("python -W ignore trial.py -d mnist -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d mnist -s FedSTC -c configs/workstation -m server -i true -n 1 -r 1 -e run")
os.system("python -W ignore trial.py -d femnist -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d femnist -s FedSTC -c configs/workstation -m server -i true -r 1 -e run")
os.system("python -W ignore trial.py -d celeba -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d celeba -s FedSTC -c configs/workstation -m server -i true -r 1 -e run")
os.system("python -W ignore trial.py -d semantic140 -s FedSTC -c configs/workstation -m server -i false -r 1 -e run")
os.system("python -W ignore trial.py -d semantic140 -s FedSTC -c configs/workstation -m server -i true -r 1 -e run")
