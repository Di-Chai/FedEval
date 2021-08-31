import os

# MNIST IID
os.system("python3 -W ignore trial.py -d mnist -s FedSGD -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedAvg -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedProx -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedSTC -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedOpt -c configs/local -m local -i false -r 10 -e run")

# MNIST Non-IID
os.system("python3 -W ignore trial.py -d mnist -s FedSGD -c configs/local -m local -i true -n 1 -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedAvg -c configs/local -m local -i true -n 1 -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedProx -c configs/local -m local -i true -n 1 -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedSTC -c configs/local -m local -i true -n 1 -r 10 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedOpt -c configs/local -m local -i true -n 1 -r 10 -e run")

# FEMNIST IID
os.system("python3 -W ignore trial.py -d femnist -s FedSGD -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedAvg -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedProx -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedSTC -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedOpt -c configs/local -m local -i false -r 10 -e run")

# FEMNIST Non-IID
os.system("python3 -W ignore trial.py -d femnist -s FedSGD -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedAvg -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedProx -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedSTC -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedOpt -c configs/local -m local -i true -r 10 -e run")

# CelebA IID
os.system("python3 -W ignore trial.py -d celeba -s FedSGD -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedAvg -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedProx -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedSTC -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedOpt -c configs/local -m local -i false -r 10 -e run")

# CelebA Non-IID
os.system("python3 -W ignore trial.py -d celeba -s FedSGD -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedAvg -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedProx -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedSTC -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedOpt -c configs/local -m local -i true -r 10 -e run")

# Sent140 IID
os.system("python3 -W ignore trial.py -d semantic140 -s FedSGD -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedAvg -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedProx -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedSTC -c configs/local -m local -i false -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedOpt -c configs/local -m local -i false -r 10 -e run")

# Sent140 Non-IID
os.system("python3 -W ignore trial.py -d semantic140 -s FedSGD -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedAvg -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedProx -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedSTC -c configs/local -m local -i true -r 10 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedOpt -c configs/local -m local -i true -r 10 -e run")
