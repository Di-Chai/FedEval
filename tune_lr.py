import os
import socket

"""
Tune the learning rates for FedSTC, FedProx, and FedOpt
"""

host_name = socket.gethostname()

if host_name == "gpu05":

    os.system("python3 -W ignore trial.py -d mnist -s FedSTC -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d mnist -s FedProx -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d mnist -s FedOpt -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")

    os.system("python3 -W ignore trial.py -d femnist -s FedSTC -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d femnist -s FedProx -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d femnist -s FedOpt -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")


if host_name == "gpu06":

    os.system("python3 -W ignore trial.py -d celeba -s FedSTC -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d celeba -s FedProx -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d celeba -s FedOpt -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")

    os.system("python3 -W ignore trial.py -d semantic140 -s FedSTC -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d semantic140 -s FedProx -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")
    os.system("python3 -W ignore trial.py -d semantic140 -s FedOpt -c configs/local -m local -i false -t lr -r 5 -l log/tunelr -e run")