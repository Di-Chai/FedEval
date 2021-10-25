import os
import socket

"""
Tune the learning rates for FedSTC, FedProx, and FedOpt
"""

host_name = socket.gethostname()

if host_name == 'ministation':
    # os.system("python -W ignore trial.py -d mnist -s LocalCentral "
    #           "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python -W ignore trial.py -d femnist -s LocalCentral "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")

if host_name == 'workstation':

    os.system("sudo /home/ubuntu/.virtualenvs/chaidi/bin/python -W ignore trial.py "
              "-d celeba -s LocalCentral -c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("sudo /home/ubuntu/.virtualenvs/chaidi/bin/python -W ignore trial.py "
              "-d semantic140 -s LocalCentral -c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")

if host_name == "gpu05":
    
    os.system("python3 -W ignore trial.py -d mnist -s FedSGD "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d mnist -s FedAvg "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d mnist -s FedSTC "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d mnist -s FedProx "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d mnist -s FedOpt "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")

    os.system("python3 -W ignore trial.py -d femnist -s FedSGD "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d femnist -s FedAvg "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d femnist -s FedSTC "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d femnist -s FedProx "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d femnist -s FedOpt "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")


if host_name == "gpu06":

    # os.system("python3 -W ignore trial.py -d celeba -s FedSGD "
    #           "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d celeba -s FedAvg "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d celeba -s FedSTC "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d celeba -s FedProx "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d celeba -s FedOpt "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")

    os.system("python3 -W ignore trial.py -d semantic140 -s FedSGD "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d semantic140 -s FedAvg "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d semantic140 -s FedSTC "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d semantic140 -s FedProx "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d semantic140 -s FedOpt "
              "-c configs/quickstart -m local -i false -t lr -r 1 -l log/tunelr -e run")

