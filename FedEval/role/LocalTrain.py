import os
import gc
import time
import json
import pickle
import requests
import logging
import numpy as np

from ..utils import pickle_string_to_obj


class NormalTrain(object):

    def __init__(self, model, data_dir):
        self.model = model
        self.data_dir = data_dir

    def local_train(self):
        data_files = [e for e in os.listdir(self.data_dir) if e.startswith('client') and e.endswith('.pkl')]
        data = [pickle_string_to_obj(e) for e in data_files]

        for client_data in data:
            pass

    def central_train(self):
        pass
