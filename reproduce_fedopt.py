import os

# FedOpt
os.system("python3 -W ignore trial.py -d mnist -s FedOpt -c configs/local -m local -i false -r 5 -e run")
os.system("python3 -W ignore trial.py -d mnist -s FedOpt -c configs/local -m local -i true -n 1 -r 5 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedOpt -c configs/local -m local -i false -r 5 -e run")
os.system("python3 -W ignore trial.py -d femnist -s FedOpt -c configs/local -m local -i true -r 5 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedOpt -c configs/local -m local -i false -r 5 -e run")
os.system("python3 -W ignore trial.py -d celeba -s FedOpt -c configs/local -m local -i true -r 5 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedOpt -c configs/local -m local -i false -r 5 -e run")
os.system("python3 -W ignore trial.py -d semantic140 -s FedOpt -c configs/local -m local -i true -r 5 -e run")
