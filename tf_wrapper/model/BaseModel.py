import os
import numpy as np
import tensorflow as tf

from tensorboard.backend.event_processing import event_accumulator

from tf_wrapper.train.MiniBatchTrain import MiniBatchFeedDict
from tf_wrapper.train import EarlyStopping, EarlyStoppingTTest
from tf_wrapper.preprocess import SplitData


class BaseModel(object):
    """BaseModel is the base class for many models, such as STMeta, ST-MGCN and ST_ResNet,
        you can also build your own model using this class. More information can be found in tutorial.
    Args:
        code_version: Current version of this model code, which will be used as filename for saving the model.
        model_dir: The directory to store model files. Default:'model_dir'.
        gpu_device: To specify the GPU to use. Default: '0'.
    """

    def __init__(self, inputs_shape, targets_shape, code_version, model_dir, gpu_device):

        # model input and output
        self.inputs_shape = inputs_shape
        self.targets_shape = targets_shape
        self.input = {}
        self.output = {}
        self.op = {}
        self.variable_init = None
        self.optimizer = None
        self.saver = None
        self.model = None

        self.code_version = code_version
        self.model_dir = model_dir
        self.is_build = False

        # TF Graph
        self.graph = tf.Graph()

        self.converged = False
        self.log_dir = os.path.join(self.model_dir, self.code_version)
        self.global_step = 0
        self._summary = None
        self._summary_writer = None

        self.trainable_vars = 0

        # TF Session
        self._GPU_DEVICE = gpu_device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self._GPU_DEVICE
        self._config = tf.ConfigProto()
        self._config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=self._config)

    @staticmethod
    def parse_activation(activation_name):
        try:
            return eval('tf.nn.' + activation_name)
        except Exception as e:
            print('Error', e)
            exit(1)

    @staticmethod
    def conv2d(x, conv_filter, activation, stride, padding, name, trainable=True):
        w = tf.compat.v1.get_variable(name + '/kernel', conv_filter, trainable=trainable)
        y = tf.nn.conv2d(x, w, strides=stride, padding=padding)
        w_bias = tf.compat.v1.get_variable(name + '/bias', y.get_shape().as_list()[-1:], trainable=trainable)
        if activation is not None:
            return activation(y + w_bias, 'ReLU/' + name)
        else:
            return tf.add(y, w_bias, name=name)

    @staticmethod
    def dense(x, units, activation, name, trainable=True):
        w = tf.compat.v1.get_variable(name + '/kernel', (x.get_shape().as_list()[-1], units), trainable=trainable)
        b = tf.compat.v1.get_variable(name + '/bias', (units,), trainable=trainable)
        if activation is not None:
            return activation(tf.matmul(x, w) + b, name=name)
        else:
            return tf.add(tf.matmul(x, w), b, name=name)

    @staticmethod
    def cross_entropy_loss(labels, logits):
        return tf.reduce_sum(tf.multiply(labels, -tf.log(logits)), axis=-1)

    def forward(self, inputs, targets, trainable):
        """You need to override this function"""
        return None

    def build(self, init_vars=True, max_to_keep=5, save_hist=False):

        with self.graph.as_default():

            input_tensor = {}
            for input_name in self.inputs_shape:
                input_tensor[input_name] = \
                    tf.compat.v1.placeholder(tf.float32, [None] + list(self.inputs_shape[input_name]), name=input_name)
                self.input[input_name] = input_tensor[input_name].name

            target_tensor = {}
            for target_name in self.targets_shape:
                target_tensor[target_name] = \
                    tf.compat.v1.placeholder(tf.float32, [None] + list(self.targets_shape[target_name]),
                                             name=target_name)
                self.input[target_name] = target_tensor[target_name].name

            self.forward(inputs=input_tensor, targets=target_tensor, trainable=True)

            self.build_essential(init_vars=init_vars, max_to_keep=max_to_keep, save_hist=save_hist)

        self.is_build = True

    def build_essential(self, init_vars=True, max_to_keep=5, save_hist=False):
        """
        Args
            init_vars(bool): auto init the parameters if set to True, else no parameters will be initialized.
            max_to_keep: max file to keep, which equals to max_to_keep in tf.train.Saver.
        """
        with self.graph.as_default():
            ####################################################################
            # Add summary, variable_init and summary
            # The variable name of them are fixed
            self.trainable_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            self.saver = tf.train.Saver(max_to_keep=max_to_keep)
            if self.variable_init is None:
                self.variable_init = tf.global_variables_initializer()
            if save_hist:
                self._summary_writer = tf.summary.FileWriter(self.log_dir)
                self._summary = self._summary_histogram().name
            ####################################################################
        if init_vars:
            self.session.run(self.variable_init)

    def add_summary(self, name, value, global_step):
        value_record = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
        self._summary_writer.add_summary(value_record, global_step)

    def _summary_histogram(self):
        with self.graph.as_default():
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
        self._summary_writer.add_graph(self.graph)
        return tf.summary.merge_all()

    def _run(self, feed_dict, output_names, op_names):
        feed_dict_tf = {}
        for name, value in feed_dict.items():
            if value is not None:
                feed_dict_tf[self.graph.get_tensor_by_name(self.input[name])] = value

        output_tensor_list = [self.graph.get_tensor_by_name(self.output[name]) for name in output_names]
        output_tensor_list += [self.graph.get_operation_by_name(self.op[name]) for name in op_names]

        outputs = self.session.run(output_tensor_list, feed_dict=feed_dict_tf)

        return {output_names[i]: outputs[i] for i in range(len(output_names))}

    def _get_feed_dict(self, **kwargs):
        return kwargs

    def fit(self, train_data, val_data=None, validate_ratio=None,
            train_sample_size=None, val_sample_size=None, output_names=('loss',),
            op_names=('train_op',), evaluate_loss_name='loss', batch_size=None, max_epoch=10000, shuffle_data=True,
            early_stop_method='t-test', early_stop_length=10, early_stop_patience=0.1,
            verbose=True, save_model=False, save_model_name=None, auto_load_model=True,
            return_outputs=False):

        """
        Args:
            train_sample_size: int, the sequence length which is use in mini-batch training
            output_names: list, [output_tensor1_name, output_tensor1_name, ...]
            op_names: list, [operation1_name, operation2_name, ...]
            evaluate_loss_name: str, should be on of the output_names, evaluate_loss_name was use in
                                       early-stopping
            batch_size: int, default 64, batch size
            max_epoch: int, default 10000, max number of epochs
            validate_ratio: float, default 0.1, the ration of data that will be used as validation dataset
            shuffle_data: bool, default True, whether shuffle data in mini-batch train
            early_stop_method: should be 't-test' or 'naive', both method are explained in train.EarlyStopping
            early_stop_length: int, must provide when early_stop_method='t-test'
            early_stop_patience: int, must provide when early_stop_method='naive'
            verbose: Bool, flag to print training information or not
            save_model: Bool, flog to save model or not
            save_model_name: String, filename for saving the model, which will overwrite the code_version.
            auto_load_model: Bool, the "fit" function will automatically load the model from disk, if exists,
                before the training. Set to False to disable the auto-loading.
            return_outputs: Bool, set True to return the training log, otherwise nothing will be returned
        """

        if auto_load_model:
            try:
                self.load(self.code_version)
                print('Found model in disk')
                if self.converged:
                    print('Model converged, stop training')
                    return
                else:
                    print('Model not converged, continue at step', self.global_step)
                    start_epoch = self.global_step
            except FileNotFoundError:
                print('No model found, start training')
                start_epoch = 0
        else:
            start_epoch = 0
            print('Not loading model from disk')

        if val_data is None and (validate_ratio is None or not 0 <= validate_ratio < 1):
            raise ValueError('validate_ratio should between [0, 1), given', validate_ratio)

        if evaluate_loss_name not in output_names:
            raise ValueError('evaluate_loss_name not shown in', output_names)

        if len(op_names) == 0:
            raise ValueError('No operation given')
        else:
            print('Running Operation', op_names)

        if self._summary_writer is None:
            self._summary_writer = tf.summary.FileWriter(self.log_dir)

        # Get feed_dict
        feed_dict = self._get_feed_dict(**train_data)

        if train_sample_size is None:
            train_sample_size = [v.shape[0] for _, v in feed_dict.items()]
            assert min(train_sample_size) == max(train_sample_size)
            train_sample_size = train_sample_size[0]

        if batch_size is None:
            batch_size = train_sample_size

        if val_data is not None:
            val_feed_dict = self._get_feed_dict(**val_data)
            train_feed_dict = feed_dict

            if val_sample_size is None:
                val_sample_size = [v.shape[0] for _, v in val_feed_dict.items()]
                assert min(val_sample_size) == max(val_sample_size)
                val_sample_size = val_sample_size[0]

            train_sequence_length = train_sample_size
            val_sequence_len = val_sample_size

        else:
            if validate_ratio == 0:
                train_feed_dict = feed_dict
                val_feed_dict = feed_dict
            else:
                # Split data into train-data and validation data
                train_feed_dict, val_feed_dict = SplitData.split_feed_dict(
                    feed_dict,
                    sequence_length=train_sample_size,
                    ratio_list=[1 - validate_ratio, validate_ratio]
                )

            train_sequence_length = int(train_sample_size * (1 - validate_ratio))
            val_sequence_len = train_sample_size - train_sequence_length

        # build mini-batch data source on train-data
        train_dict_mini_batch = MiniBatchFeedDict(feed_dict=train_feed_dict,
                                                  sequence_length=train_sequence_length,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_data)

        # record the best result of "evaluate_loss_name"
        best_record = None
        # init early stopping object
        if early_stop_method.lower() == 't-test':
            early_stop = EarlyStoppingTTest(length=early_stop_length, p_value_threshold=early_stop_patience)
        else:
            early_stop = EarlyStopping(patience=int(early_stop_patience))

        # start mini-batch training
        summary_output = []
        for epoch in range(start_epoch, max_epoch):
            train_output_list = []
            for i in range(train_dict_mini_batch.num_batch):
                # train
                train_output = self._run(feed_dict=train_dict_mini_batch.get_batch(),
                                         output_names=output_names,
                                         op_names=op_names)
                train_output_list.append(train_output)

            # validation
            val_output = self.predict(val_feed_dict, output_names=output_names,
                                      sample_size=val_sequence_len,
                                      cache_volume=batch_size)

            # Here we only care about the evaluate_loss_value
            evaluate_loss_value = np.mean(val_output[evaluate_loss_name])

            # Add Summary
            tmp_summary = {}
            for name in output_names:
                if save_model:
                    self.add_summary(name='train_' + name,
                                     value=np.mean([e[name] for e in train_output_list]),
                                     global_step=epoch)
                    self.add_summary(name='val_' + name, value=np.mean(val_output[name]), global_step=epoch)
                # print training messages
                if verbose:
                    print('Epoch %s:' % epoch,
                          'train_' + name, np.mean([e[name] for e in train_output_list]),
                          'val_' + name, np.mean(val_output[name]))
                tmp_summary['train_' + name] = np.mean([e[name] for e in train_output_list])
                tmp_summary['val_' + name] = np.mean(val_output[name])
            summary_output.append(tmp_summary)

            # manual_summary the histograms
            if save_model:
                self.manual_summary(global_step=epoch)

            if early_stop.stop(evaluate_loss_value):
                if save_model:
                    self._log('Converged')
                break

            # save the model if evaluate_loss_value is smaller than best_record
            if (best_record is None or evaluate_loss_value < best_record) and save_model:
                best_record = evaluate_loss_value
                self.save(save_model_name or self.code_version, epoch)

        if return_outputs:
            return summary_output

    def predict(self, test_data=None, sample_size=None, output_names=('prediction',), cache_volume=64):

        '''
        Args:
            output_names: list, [output_tensor_name1, output_tensor_name2, ...]
            sequence_length: int, the length of sequence, which is use in mini-batch training
            cache_volume: int, default 64, we need to set cache_volume if the cache can not hold
                                 the whole validation dataset
            :return: outputs_dict: dict, like {output_tensor1_name: output_tensor1_value, ...}
        '''

        # Get feed_dict
        if test_data is not None:
            feed_dict = self._get_feed_dict(**test_data)
        else:
            feed_dict = {}

        if sample_size is None:
            sample_size = [v.shape[0] for _, v in feed_dict.items()]
            assert min(sample_size) == max(sample_size)
            sample_size = sample_size[0]

        if cache_volume and sample_size:
            # storing the prediction result
            outputs_list = []
            outputs_dict = {}
            for i in range(0, sample_size, cache_volume):
                tmp_output = self._run({key: value[i:i + cache_volume] if len(value) == sample_size else value
                                        for key, value in feed_dict.items()},
                                       output_names, op_names=[])
                outputs_list.append(tmp_output)
            # stack the output together
            for key in outputs_list[0]:
                outputs_dict[key] = np.concatenate([e[key] for e in outputs_list], axis=0)
        else:
            outputs_dict = self._run(feed_dict, output_names, op_names=[])

        return outputs_dict

    def get_weights(self):
        with self.graph.as_default():
            target_vars = tf.all_variables()
            return {var.name: self.session.run(var) for var in target_vars}

    def set_weights(self, value_dict):
        with self.graph.as_default():
            for variable in tf.all_variables():
                if variable.name in value_dict:
                    variable.load(value_dict[variable.name], self.session)
                    print('Load', variable.name)

    def manual_summary(self, global_step=None):
        self._summary_writer.add_summary(self.session.run(self.graph.get_tensor_by_name(self._summary)),
                                         global_step=global_step)

    def _log(self, text):
        save_dir_subscript = os.path.join(self.log_dir, self.code_version)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        with open(os.path.join(save_dir_subscript, 'log.txt'), 'a+', encoding='utf-8') as f:
            f.write(text + '\n')

    def _get_log(self):
        save_dir_subscript = os.path.join(self.log_dir, self.code_version)
        if os.path.isfile(os.path.join(save_dir_subscript, 'log.txt')):
            with open(os.path.join(save_dir_subscript, 'log.txt'), 'r', encoding='utf-8') as f:
                return [e.strip('\n') for e in f.readlines()]
        else:
            return []

    def save(self, subscript, global_step):
        """
        Args:
            subscript: String, subscript will be appended to the code version as the model filename,
                and save the corresponding model using this filename
            global_step: Int, current training steps
        """
        save_dir_subscript = os.path.join(self.log_dir, subscript)
        # delete if exist
        # if os.path.isdir(save_dir_subscript):
        #     shutil.rmtree(save_dir_subscript, ignore_errors=True)
        if os.path.isdir(save_dir_subscript) is False:
            os.makedirs(save_dir_subscript)
        self.saver.save(sess=self.session, save_path=os.path.join(save_dir_subscript, subscript),
                        global_step=global_step)

    def load(self, subscript):
        """
        Args:
            subscript: String, subscript will be appended to the code version as the model file name,
                and load the corresponding model using this filename
        """
        save_dir_subscript = os.path.join(self.log_dir, subscript)
        if len(os.listdir(save_dir_subscript)) == 0:
            print('model Not Found')
            raise FileNotFoundError(subscript, 'model not found')
        else:
            meta_file = [e for e in os.listdir(save_dir_subscript) if e.startswith(subscript) and e.endswith('.meta')]
            self.global_step = max([int(e.split('.')[0].split('-')[-1]) for e in meta_file])
            self.saver.restore(sess=self.session,
                               save_path=os.path.join(save_dir_subscript, subscript + '-%s' % self.global_step))
            self.global_step += 1
            # parse the log-file
            log_list = self._get_log()
            for e in log_list:
                if e.lower() == 'converged':
                    self.converged = True

    def close(self):
        """
        Close the session, release memory.
        """
        self.session.close()

    def load_event_scalar(self, scalar_name='val_loss'):
        """
        Args:
            scalar_name: load the corresponding scalar name from tensorboard-file,
                e.g. load_event_scalar('val_loss)
        """
        event_files = [e for e in os.listdir(self.log_dir) if e.startswith('events.out')]
        result = []
        for f in event_files:
            ea = event_accumulator.EventAccumulator(os.path.join(self.log_dir, f))
            ea.Reload()
            if scalar_name in ea.scalars.Keys():
                result += [[e.wall_time, e.step, e.value] for e in ea.scalars.Items(scalar_name)]
        return result


class FMLModel(BaseModel):

    def __init__(self, inputs_shape, targets_shape,
                 code_version='MLP', model_dir='model_dir', gpu_device='-1'):

        self.inputs_shape = inputs_shape
        self.targets_shape = targets_shape

        super(FMLModel, self).__init__(
            inputs_shape=inputs_shape, targets_shape=targets_shape,
            code_version=code_version, model_dir=model_dir, gpu_device=gpu_device
        )

    def build_attack(self, gradient_names, init_vars=True, max_to_keep=5, batch_size=None,):

        def str_match_list(v, vl):
            for e in vl:
                if e in v:
                    return True
            return False

        with self.graph.as_default():

            input_tensor = {}
            for input_name in self.inputs_shape:
                input_tensor[input_name] = \
                    tf.Variable(
                        tf.random.normal(
                            [batch_size] + list(self.inputs_shape[input_name]),
                            dtype=tf.float32),
                        name=input_name
                    )
                self.output[input_name] = input_tensor[input_name].name

            target_tensor = {}
            target_tensor_raw = {}
            for target_name in self.targets_shape:
                target_tensor_raw[target_name] = tf.Variable(tf.random.normal(
                    [batch_size] + list(self.targets_shape[target_name]),
                    dtype=tf.float32),
                    name=target_name)
                target_tensor[target_name] = tf.nn.softmax(target_tensor_raw[target_name], axis=-1)
                self.output[target_name] = target_tensor[target_name].name

            loss = self.forward(inputs=input_tensor, targets=target_tensor, trainable=False)

            assert loss is not None

            gradients = [[e, tf.gradients(loss, e)[0]] for e in tf.global_variables()
                         if str_match_list(e.name, gradient_names)]

            optimizer = tf.train.GradientDescentOptimizer(self.lr)

            attack_loss = []
            for v, g in gradients:
                tmp_g = tf.compat.v1.placeholder(tf.float32, g.shape, g.name.split(':')[0])
                self.input[v.name] = tmp_g.name
                self.output[v.name] = g.name
                self.output[v.name + '_y'] = tf.gradients(tf.reduce_sum(tf.square(g)), target_tensor['y'])[0].name
                assert g.shape == tmp_g.shape
                attack_loss.append(tf.reduce_sum(tf.square(g - tmp_g)))

            attack_loss = tf.reduce_sum(attack_loss)

            mask_tensor_x = tf.concat([tf.zeros([1] + list(self.inputs_shape['x'])),
                                      tf.ones([1] + list(self.inputs_shape['x']))], axis=0)

            mask_tensor_y = tf.concat([tf.zeros([1] + list(self.targets_shape['y'])),
                                       tf.ones([1] + list(self.targets_shape['y']))], axis=0)

            batch_index = tf.compat.v1.placeholder(tf.int32, [batch_size, ], name='batch_index')
            self.input['batch_index'] = batch_index.name
            update_mask_x = tf.gather(mask_tensor_x, batch_index, axis=0)
            update_mask_y = tf.gather(mask_tensor_y, batch_index, axis=0)

            self.output['attack_loss'] = attack_loss.name
            self.output['loss'] = loss.name

            optimizer_gradients = optimizer.compute_gradients(attack_loss, [input_tensor['x'], target_tensor_raw['y']])
            optimizer_gradients = [list(e) for e in optimizer_gradients]

            optimizer_gradients[0][0] = tf.multiply(update_mask_x, optimizer_gradients[0][0])
            optimizer_gradients[1][0] = tf.multiply(update_mask_y, optimizer_gradients[1][0])

            attack_train_op = optimizer.apply_gradients(optimizer_gradients)
            self.op['attack_train_op'] = attack_train_op.name

            ########################
            # TMP
            self.output['grad_x'] = tf.gradients(attack_loss, input_tensor['x'])[0].name
            self.output['grad_y'] = tf.gradients(attack_loss, target_tensor['y'])[0].name
            ########################

            super(FMLModel, self).build_essential(init_vars=init_vars, max_to_keep=max_to_keep)
