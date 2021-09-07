import json
import os
import pickle

import numpy as np
import tensorflow as tf

from ..config.configuration import ConfigurationManager
from ..role import Role
from .FedAvg import FedAvg


class LocalCentral(FedAvg):

    def __init__(self):
        super().__init__()

        cfg_mgr = ConfigurationManager()
        client_num = cfg_mgr.runtime_config.client_num
        data_dir = cfg_mgr.data_config.dir_name
        if self.role == Role.Server:
            x_train = []
            y_train = []
            x_val = []
            y_val = []
            x_test = []
            y_test = []
            for client_id in range(client_num):
                with open(os.path.join(data_dir, 'client_%s.pkl' % client_id), 'rb') as f:
                    data = pickle.load(f)
                x_train.append(data['x_train'])
                y_train.append(data['y_train'])
                x_val.append(data['x_val'])
                y_val.append(data['y_val'])
                x_test.append(data['x_test'])
                y_test.append(data['y_test'])
            self.train_data = {'x': np.array(x_train), 'y': np.array(y_train)}
            self.val_data = {'x': np.array(x_val), 'y': np.array(y_val)}
            self.test_data = {'x': np.array(x_test), 'y': np.array(y_test)}
            
            self.train_data_size = len(self.train_data['x'])
            self.val_data_size = len(self.val_data['x'])
            self.test_data_size = len(self.test_data['x'])
    
    def host_exit_job(self, host):
        train_loss, _ = self.fit_on_local_data()
        evaluate_results = self.local_evaluate()
        host.result_json.update({'central_train': evaluate_results})
        with open(os.path.join(host.log_dir, 'results.json'), 'w') as f:
            json.dump(host.result_json, f)

    def update_host_params(self, client_params, aggregate_weights):
        return None

    def set_host_params_to_local(self, host_params, current_round):
        pass

    def fit_on_local_data(self):
        self.local_params_pre = self._retrieve_local_params()
        mdl_cfg = ConfigurationManager().model_config
        train_log = self.ml_model.fit(
            **self.train_data, batch_size=mdl_cfg.B,
            epochs=mdl_cfg.E,
            validation_data=(self.val_data['x'], self.val_data['y']),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=ConfigurationManager().tolerance_num,
                    restore_best_weights=True
                )]
        )
        train_loss = train_log.history['loss'][-1]
        self.local_params_cur = self._retrieve_local_params()
        return train_loss, self.train_data_size
