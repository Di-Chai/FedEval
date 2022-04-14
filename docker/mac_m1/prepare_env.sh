#!/usr/bin/env bash


conda install -y -c apple tensorflow-deps==2.6.0
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal

pip install -r requirements.txt -i https://pypi.douban.com/simple/
