#!/bin/bash

pythonV=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')

if [[ "$pythonV" == "3.8" ]]
then 
    echo "Valid Python Version"
else
    echo "Get Python $pythonV"
    echo "Invalid Version"
    apt install -y python3.8 python3-pip --fix-missing
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$pythonV 1
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
fi