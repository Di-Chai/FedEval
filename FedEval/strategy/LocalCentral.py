import os
import json
import pickle
import numpy as np
import tensorflow as tf
from .FedAvg import FedAvg
from role import Role


class LocalCentral(FedAvg):

    def __init__(self, role: Role, data_config, model_config, runtime_config):
        super().__init__(role, data_config, model_config, runtime_config)

        if self.role == Role.Server:
            x_train = []
            y_train = []
            x_val = []
            y_val = []
            x_test = []
            y_test = []
            for client_id in range(self.runtime_config['server']['num_clients']):
                with open(os.path.join(self.data_config['data_dir'], 'client_%s.pkl' % client_id), 'rb') as f:
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
        train_log = self.ml_model.fit(
            **self.train_data, batch_size=self.model_config['FedModel']['B'],
            epochs=self.model_config['FedModel']['E'],
            validation_data=(self.val_data['x'], self.val_data['y']),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=self.model_config['FedModel']['num_tolerance'],
                    restore_best_weights=True
                )]
        )
        train_loss = train_log.history['loss'][-1]
        self.local_params_cur = self._retrieve_local_params()
        return train_loss, self.train_data_size
