import os
import socket

host_name = socket.gethostname()

if host_name == "gpu06":
    
    os.system("python -W ignore trial.py -d mnist -s FedSGD "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedAvg "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedSTC "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedProx "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedOpt "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")

    os.system("python -W ignore trial.py -d mnist -s FedSGD "
              "-c configs/quickstart -m local -i true -n 1 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedAvg "
              "-c configs/quickstart -m local -i true -n 1 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedSTC "
              "-c configs/quickstart -m local -i true -n 1 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedProx "
              "-c configs/quickstart -m local -i true -n 1 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedOpt "
              "-c configs/quickstart -m local -i true -n 1 -r 10 -l log/jmlr -e run")

    os.system("python -W ignore trial.py -d mnist -s FedSGD "
              "-c configs/quickstart -m local -i true -n 2 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedAvg "
              "-c configs/quickstart -m local -i true -n 2 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedSTC "
              "-c configs/quickstart -m local -i true -n 2 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedProx "
              "-c configs/quickstart -m local -i true -n 2 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedOpt "
              "-c configs/quickstart -m local -i true -n 2 -r 10 -l log/jmlr -e run")

    os.system("python -W ignore trial.py -d mnist -s FedSGD "
              "-c configs/quickstart -m local -i true -n 3 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedAvg "
              "-c configs/quickstart -m local -i true -n 3 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedSTC "
              "-c configs/quickstart -m local -i true -n 3 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedProx "
              "-c configs/quickstart -m local -i true -n 3 -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d mnist -s FedOpt "
              "-c configs/quickstart -m local -i true -n 3 -r 10 -l log/jmlr -e run")
    
if host_name == 'gpu01':

    os.system("python -W ignore trial.py -d femnist -s FedSGD "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedAvg "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedSTC "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedProx "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedOpt "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")

    os.system("python -W ignore trial.py -d femnist -s FedSGD "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedAvg "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedSTC "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedProx "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d femnist -s FedOpt "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")


if host_name == 'gpu02':

    os.system("python -W ignore trial.py -d semantic140 -s FedSGD "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedAvg "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedSTC "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedProx "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedOpt "
              "-c configs/quickstart -m local -i false -r 10 -l log/jmlr -e run")

    os.system("python -W ignore trial.py -d semantic140 -s FedSGD "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedAvg "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedSTC "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedProx "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")
    os.system("python -W ignore trial.py -d semantic140 -s FedOpt "
              "-c configs/quickstart -m local -i true -r 10 -l log/jmlr -e run")

if host_name == 'mac':

    os.system("python -W ignore trial.py -d femnist -s LocalCentral "
              "-c configs/quickstart -m local -i false -r 1 -l log/jmlr -e run")
