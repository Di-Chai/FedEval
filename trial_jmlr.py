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
    
