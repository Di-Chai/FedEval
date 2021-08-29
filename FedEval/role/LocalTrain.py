import os
import numpy as np
import tensorflow as tf

from ..run import generate_data
from ..utils import ParamParser


class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)


class NormalTrain(object):

    def __init__(self, data_config, model_config, runtime_config):

        # Config the GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.clients_data = generate_data(
            data_config=data_config, model_config=model_config, runtime_config=runtime_config, save_file=False
        )
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = [], [], [], [], [], []

        self.data_config = data_config
        self.model_config = model_config
        self.runtime_config = runtime_config

        for e in self.clients_data:
            self.x_train.append(e['x_train'])
            self.y_train.append(e['y_train'])
            self.x_val.append(e['x_val'])
            self.y_val.append(e['y_val'])
            self.x_test.append(e['x_test'])
            self.y_test.append(e['y_test'])

        self.x_train = np.concatenate(self.x_train)
        self.y_train = np.concatenate(self.y_train)
        self.x_val = np.concatenate(self.x_val)
        self.y_val = np.concatenate(self.y_val)
        self.x_test = np.concatenate(self.x_test)
        self.y_test = np.concatenate(self.y_test)

        self.param_parser = ParamParser(
            data_config=data_config, model_config=model_config, runtime_config=runtime_config
        )
        self.ml_model = self.param_parser.parse_model()
        self.init_ml_model_params = self.ml_model.get_weights()

    def local_train(self):
        evaluation_results = []
        client_counter = 1
        for each_client in self.clients_data:
            print('Client', client_counter, 'local training')
            client_counter += 1
            x_train = each_client['x_train']
            y_train = each_client['y_train']
            x_val = each_client['x_val']
            y_val = each_client['y_val']
            x_test = each_client['x_test']
            y_test = each_client['y_test']
            self.ml_model.set_weights(self.init_ml_model_params)
            self.ml_model.fit(
                x_train, y_train, batch_size=self.model_config['FedModel']['B'],
                epochs=self.model_config['FedModel']['max_rounds'], validation_data=(x_val, y_val),
                callbacks=[EarlyStopping(
                    monitor='val_loss', restore_best_weights=True,
                    patience=self.model_config['FedModel']['num_tolerance'])]
            )
            test_result = self.ml_model.evaluate(x=x_test, y=y_test)
            evaluation_results.append(test_result)
        return evaluation_results

    def central_train(self):
        print('Central Training')
        self.ml_model.set_weights(self.init_ml_model_params)
        self.ml_model.fit(
            self.x_train, self.y_train, batch_size=self.model_config['FedModel']['B'],
            epochs=self.model_config['FedModel']['max_rounds'], validation_data=(self.x_val, self.y_val),
            callbacks=[
                EarlyStopping(
                    monitor='val_loss', patience=self.model_config['FedModel']['num_tolerance'],
                    restore_best_weights=True
                )])
        test_result = self.ml_model.evaluate(x=self.x_test, y=self.y_test)
        return test_result

    def run(self, file_name='debug.csv'):
        central_train_results = self.central_train()
        local_train_results = self.local_train()

        local_train_results = np.average(
            local_train_results, axis=0,
            weights=[len(e['x_train']) / len(self.x_train) for e in self.clients_data]
        )
        headers = ['dataset', 'lr']
        headers += (['local_' + e for e in self.ml_model.metrics_names] +
                    ['central_' + e for e in self.ml_model.metrics_names])

        results = [self.data_config['dataset'], self.ml_model.optimizer.lr.numpy()]
        results += local_train_results.tolist()
        results += list(central_train_results)

        if os.path.isfile(file_name) is False:
            f = open(file_name, 'w')
            f.write(', '.join(headers) + '\n')
        else:
            f = open(file_name, 'a+')

        f.write(', '.join([str(e) for e in results]) + '\n')
        f.close()
