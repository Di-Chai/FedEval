#!/usr/bin/env bash

# GPU6
python -W ignore trial.py -d celeba -s FedAvg -c configs/quickstart -m local -i true -t lr -r 5 -l log/tunelr_jmlr -e run
# GPU1
python -W ignore trial.py -d femnist -s FedAvg -c configs/quickstart -m local -i true -t lr -r 5 -l log/tunelr_jmlr -e run
# GPU5
python -W ignore trial.py -d semantic140 -s FedAvg -c configs/quickstart -m local -i true -t lr -r 5 -l log/tunelr_jmlr -e run
# GPU2
python -W ignore trial.py -d shakespeare -s FedAvg -c configs/quickstart -m local -i true -t lr -r 5 -l log/tunelr_jmlr -e run
