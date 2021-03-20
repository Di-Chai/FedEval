import numpy as np
import tensorflow as tf

from scipy.sparse import csr_matrix
from tf_wrapper.model import BaseModel, FMLModel


def str_hit(target, filter_list):
    for value in filter_list:
        if value in target:
            return True
    return False


# def max_p(value_list, p=0.1, error=0.0001):
#
#     target = np.abs(value_list.reshape([-1]))
#     target_len = int(len(target) * p)
#
#     max_v = np.max(target)
#     min_v = np.min(target)
#     cut_v = np.median(target)
#
#     while abs(len(target) - target_len) > int(target_len * error):
#         if np.sum(target > cut_v) > target_len:
#             target = np.compress(target > cut_v, target)
#             min_v = cut_v
#             cut_v = (max_v + cut_v) / 2
#         else:
#             max_v = cut_v
#             cut_v = (min_v + cut_v) / 2
#     return cut_v


def max_p(value_list, p=0.1):
    target = np.sort(np.abs(value_list).reshape([-1]))
    target_len = int(len(target) * p)
    return target[-target_len]


def parse_strategy_and_compress(weights_old, weights_new, upload_sparse, upload_strategy):
    if upload_strategy.lower() == 'no-compress':
        return weights_new
    else:
        assert upload_sparse
        trainable_g_negative = {key: weights_new['trainable'][key] - weights_old['trainable'][key]
                                for key in weights_new['trainable']}
        if len(weights_old['optimizer']) == 0:
            optimizer_g_negative = weights_new['optimizer']
        else:
            optimizer_g_negative = {key: weights_new['optimizer'][key] - weights_old['optimizer'][key]
                                    for key in weights_new['optimizer']}
        for key in trainable_g_negative:
            t_g_v = max_p(trainable_g_negative[key], p=upload_sparse)
            trainable_g_negative[key][np.where(np.abs(trainable_g_negative[key]) < t_g_v)] = 0
            trainable_g_negative[key] = trainable_g_negative[key].reshape([-1, ])
            trainable_g_negative[key] = csr_matrix(trainable_g_negative[key])

        for key in optimizer_g_negative:
            t_g_v = max_p(optimizer_g_negative[key], p=upload_sparse)
            optimizer_g_negative[key][np.where(np.abs(optimizer_g_negative[key]) < t_g_v)] = 0
            optimizer_g_negative[key] = optimizer_g_negative[key].reshape([-1, ])
            optimizer_g_negative[key] = csr_matrix(optimizer_g_negative[key])

        return {'trainable': trainable_g_negative, 'optimizer': optimizer_g_negative}


def recover_to_weights(weights, new_g_negative, upload_strategy):
    if upload_strategy.lower() == 'no-compress':
        return new_g_negative
    else:
        for key in weights['trainable']:
            weights['trainable'][key] += \
                new_g_negative['trainable'][key].toarray().reshape(weights['trainable'][key].shape)

        for key in new_g_negative['optimizer']:
            received_value = new_g_negative['optimizer'][key].toarray().reshape(weights['optimizer'][key].shape)
            weights['optimizer'][key][np.where(received_value > 0)] = 0
            weights['optimizer'][key] += received_value
        return weights


class TFModel(FMLModel):

    def get_optimizer_weights(self, dismiss_var_names=()):
        with self.graph.as_default():
            if self.optimizer is None:
                raise ValueError('Model has no optimizer, please define one in the forward function')
            optimizer_weight = {
                var.cid: self.session.run(var) for var in tf.all_variables()
                if str_hit(var.cid, [self.optimizer.get_name()]) and not str_hit(var.cid, dismiss_var_names)}
            return optimizer_weight

    def get_optimizer_shape(self):
        optimizer_weights = self.get_optimizer_weights()
        return {key: optimizer_weights[key].shape for key in optimizer_weights}

    def get_trainable_weights(self, dismiss_var_names=()):
        with self.graph.as_default():
            trainable_weights = {var.cid: self.session.run(var) for var in tf.trainable_variables()
                                 if not str_hit(var.cid, dismiss_var_names)}
            return trainable_weights
                
    def train_one_round(self, training_data, batch_size=None, epoch=1,):
        train_summary = self.fit(
            training_data,
            output_names=('loss', 'accuracy'),
            op_names=('train_op',),
            evaluate_loss_name='loss',
            max_epoch=epoch,
            batch_size=batch_size,
            validate_ratio=0,
            save_model=False,
            return_outputs=True,
            verbose=False
        )
        train_loss = np.mean(train_summary[-1]['train_loss']).astype(float)
        return train_loss

    def evaluate(self, eval_data):
        train_summary = self.predict(eval_data, output_names=('loss', 'accuracy'))
        train_summary = {key: np.mean(value).astype(float) for key, value in train_summary.items()}
        train_summary['default'] = train_summary['loss']
        return train_summary


class KerasModel:

    def __init__(self, inputs_shape,
                 targets_shape,
                 code_version='MobileNet',
                 model_dir='log',
                 gpu_device='-1'):

        self.inputs_shape = inputs_shape
        self.targets_shape = targets_shape
        self.code_version = code_version
        self.model_dir = model_dir
        self.model = None
        self.gpu_device = gpu_device
        self.is_build = False

    def build(self):
        # Override this function
        self.is_build = True

    def get_optimizer_weights(self):
        optimizer_weights_name = [e.cid for e in self.model.optimizer.weights]
        optimizer_weights_value = self.model.optimizer.get_weights()
        optimizer_weights = {optimizer_weights_name[e]: optimizer_weights_value[e]
                             for e in range(len(optimizer_weights_name))
                             if not str_hit(optimizer_weights_name[e], ['iter'])}
        return optimizer_weights

    def get_optimizer_shape(self):
        optimizer_weights = self.get_optimizer_weights()
        return {key: optimizer_weights[key].shape for key in optimizer_weights}

    def get_weights(self, upload_name_filter=()):
        all_weights_name = [e.cid for e in self.model.weights]
        all_weights_value = self.model.get_weights()
        trainable_weights_name = [e.cid for e in self.model.trainable_weights]
        trainable_weights = {all_weights_name[e]: all_weights_value[e] for e in range(len(all_weights_name))
                             if all_weights_name[e] in trainable_weights_name and
                             not str_hit(all_weights_name[e], upload_name_filter)}
        optimizer_weights = self.get_optimizer_weights()
        optimizer_weights = {key: value for key, value in optimizer_weights.items()
                             if not str_hit(key, upload_name_filter)}
        return {'trainable': trainable_weights, 'optimizer': optimizer_weights}

    def set_weights(self, value_dict):
        all_weights_name = [e.cid for e in self.model.weights]
        all_weights_value = self.model.get_weights()
        for i in range(len(all_weights_name)):
            if all_weights_name[i] in value_dict['trainable']:
                all_weights_value[i] = value_dict['trainable'][all_weights_name[i]]
        optimizer_weights_name = [e.cid for e in self.model.optimizer.weights]
        optimizer_weights_value = self.model.optimizer.get_weights()
        for i in range(len(optimizer_weights_name)):
            if optimizer_weights_name[i] in value_dict['optimizer']:
                optimizer_weights_value[i] = value_dict['optimizer'][optimizer_weights_name[i]]
        self.model.set_weights(all_weights_value)
        self.model.optimizer.set_weights(optimizer_weights_value)

    def train_one_round(self, training_data, batch_size, epoch=1):
        train_summary = self.model.fit(x=training_data['x'], y=training_data['y'],
                                       batch_size=batch_size, epochs=epoch, verbose=0)
        train_loss = train_summary.history['loss'][-1]
        return float(train_loss)

    def evaluate(self, eval_data):
        eval_result = self.model.evaluate(x=eval_data['x'], y=eval_data['y'], batch_size=8)
        eval_summary = {'loss': float(eval_result[0]), 'accuracy': float(eval_result[1])}
        eval_summary['default'] = eval_summary['loss']
        return eval_summary

    def close(self):
        tf.keras.backend.clear_session()