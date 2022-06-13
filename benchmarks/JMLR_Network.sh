#!/usr/bin/env bash

cd ..

export FED_EVAL_REPEAT=1
export FED_EVAL_LOG_DIR=log/JMLRNetwork

export FED_EVAL_PYTHON=$(which python)

# SecureAggregation
$FED_EVAL_PYTHON trial.py -s SecureAggregation -r $FED_EVAL_REPEAT -l $FED_EVAL_LOG_DIR -c configs/quickstart -m local -t network -i true -n 1 -e run -d mnist
