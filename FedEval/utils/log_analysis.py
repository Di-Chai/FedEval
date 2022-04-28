import os
import json
import numpy as np

import matplotlib.pyplot as plt

from ..config import ConfigurationManager


class LogAnalysis:

    def __init__(self, log_dir):
        self.log_dir = log_dir

        self.federated_log_dirs = []
        self.log_files_central_simulate = []
        self.log_files_local_simulate = []
        self.log_files_fedsgd_simulate = []
        for ld in os.listdir(self.log_dir):
            if ld.startswith('.'):
                continue
            target_ld = os.path.join(self.log_dir, ld)
            if not os.path.isdir(target_ld):
                continue
            if 'Server' in os.listdir(target_ld):
                self.federated_log_dirs.append(target_ld)
            else:
                files = [os.path.join(target_ld, e) for e in os.listdir(target_ld)]
                self.log_files_central_simulate += [e for e in files if e.endswith('central_simulator.csv')]
                self.log_files_fedsgd_simulate += [e for e in files if e.endswith('fed_sgd_simulator.csv')]
                self.log_files_local_simulate += [e for e in files if e.endswith('local_simulator.csv')]

        if len(self.log_files_central_simulate) > 0:
            self.process_central_simulate_results(self.log_files_central_simulate)
        if len(self.log_files_fedsgd_simulate) > 0:
            self.process_fedsgd_simulate_results(self.log_files_fedsgd_simulate)
        if len(self.log_files_local_simulate) > 0:
            self.process_local_simulate_results(self.log_files_local_simulate)

        self.configs = []
        self.results = []
        for log in self.federated_log_dirs:
            try:
                c1, c2, c3 = ConfigurationManager.load_configs(os.path.join(log, 'Server'))
                with open(os.path.join(log, 'Server', 'results.json'), 'r') as f:
                    results = json.load(f)
                self.results.append(results)
                self.configs.append({'data_config': c1, 'model_config': c2, 'runtime_config': c3})
                print('Get log', log)
            except:
                continue

        self.omit_keys = [
            'runtime_config$$machines',
            'runtime_config$$server',
            'data_config$$random_seed'
        ]

        def check_omit(key):
            for e in self.omit_keys:
                if e in key:
                    return True
            return False

        self.key_templates = [
            [key_chain for key_chain in self.parse_dict_keys(e) if not check_omit(key_chain)]
            for e in self.configs
        ]

        self.key_templates = max(self.key_templates, key=lambda x: len(x))

        self.diff_keys = []
        for key_chain in self.key_templates:
            tmp_len = len(self.diff_keys)
            for i in range(len(self.configs)):
                for j in range(len(self.configs)):
                    if i == j:
                        continue
                    else:
                        if self.recursive_retrieve(self.configs[i], key_chain) != \
                                self.recursive_retrieve(self.configs[j], key_chain):
                            self.diff_keys.append(key_chain)
                            break
                if len(self.diff_keys) > tmp_len:
                    break

        self.csv_result_keys = [
            ['central_train$$test_accuracy', lambda x: [x] if x is None else [float(x)]],
            ['best_metric$$test_accuracy', lambda x: [float(x)]],
            ['total_time', lambda x: [int(x.split(':')[0]) * 60 + int(x.split(':')[1]) + int(x.split(':')[2]) / 60]],
            ['total_rounds', lambda x: [int(x)]],
            ['server_send', lambda x: [float(x)]],
            ['server_receive', lambda x: [float(x)]],
            ['time_detail', lambda x: eval(x)],
        ]

        self.configs_diff = self.retrieve_diff_configs()
        self.csv_results = self.parse_results()

        self.average_results = self.aggregate_csv_results()

    def plot(self, join_keys=('data_config$$dataset',), label_keys=('model_config$$FedModel$$name',)):

        num_colors = 10
        line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
        color_map = plt.get_cmap('gist_rainbow')

        aggregate_results = {}
        for i in range(len(self.configs)):
            record = self.results[i]
            join_keys_strings = '-'.join([str(self.recursive_retrieve(self.configs[i], e)) for e in join_keys])
            label_keys_strings = '-'.join([str(self.recursive_retrieve(self.configs[i], e)) for e in label_keys])
            # Find the best index
            val_loss_list = [
                record['info_each_round'][str(e + 1)]['val_loss'] for e in range(len(record['info_each_round']))
            ]
            # best_index = val_loss_list.index(min(val_loss_list))
            best_index = len(val_loss_list)
            if best_index <= 1:
                continue
            if self.configs[i]['data_config']['dataset'] == 'semantic140':
                test_acc_key = 'test_binary_accuracy'
            else:
                test_acc_key = 'test_accuracy'
            test_acc_list = [
                record['info_each_round'][str(e + 1)][test_acc_key] for e in range(len(record['info_each_round']))
            ]
            test_acc_list = test_acc_list[:best_index]
            # CommRound to Accuracy
            cr_to_acc = [e + 1 for e in range(len(record['info_each_round']))][:best_index]
            # CommAmount to Accuracy
            ca_avg_round = (record['server_send'] + record['server_receive']) / len(record['info_each_round'])
            ca_avg_round_client = ca_avg_round / self.configs[i]['runtime_config']['server'][
                'num_clients'] * 2 ** 10  # MB
            ca_to_acc = [(e + 1) * ca_avg_round_client for e in range(len(record['info_each_round']))][:best_index]
            # Time to Accuracy
            time_to_acc = [0] + [record['info_each_round'][str(e + 1)]['timestamp'] -
                                 record['info_each_round']['1']['timestamp']
                                 for e in range(1, len(record['info_each_round']))][:best_index]

            if join_keys_strings not in aggregate_results:
                aggregate_results[join_keys_strings] = {}
            if label_keys_strings not in aggregate_results[join_keys_strings]:
                aggregate_results[join_keys_strings][label_keys_strings] = []

            assert len(cr_to_acc) == len(ca_to_acc) == len(time_to_acc) == len(test_acc_list)

            aggregate_results[join_keys_strings][label_keys_strings].append(
                [cr_to_acc, ca_to_acc, time_to_acc, test_acc_list])

        def multi_to_single(data, tag):
            max_length = max([len(e) for e in data])
            max_length_index = [len(e) for e in data].index(max_length)
            for i in range(len(data)):
                if len(data[i]) != max_length:
                    if tag == 'metric':
                        data[i] = data[i] + data[max_length_index][-(max_length - len(data[i])):]
                    elif tag == 'acc':
                        data[i] = data[i] + [data[i][-1]] * (max_length - len(data[i]))
                    else:
                        raise ValueError
            single_data = np.mean(data, axis=0)
            return single_data

        def plot_one_image(key, result_key):
            fig, ax = plt.subplots(1, 3, figsize=[30, 10])
            counter = 0
            for k2 in sorted(result_key.keys()):
                line0 = ax[0].plot(result_key[k2][0], result_key[k2][-1], label=k2)
                line1 = ax[1].plot(result_key[k2][1], result_key[k2][-1], label=k2)
                line2 = ax[2].plot(result_key[k2][2], result_key[k2][-1], label=k2)
                for line in [line0, line1, line2]:
                    # line[0].set_color(color_map(counter//len(line_styles)*float(len(line_styles))/num_colors))
                    # line[0].set_linestyle(line_styles[counter % len(line_styles)])
                    line[0].set_color(color_map(float(counter % num_colors) / num_colors))
                    line[0].set_linestyle(line_styles[counter // num_colors])
                counter += 1

            x_labels = ['CR', 'CA', 'Time']
            for i in range(3):
                ax[i].legend()
                ax[i].grid()
                ax[i].set_ylabel('Accuracy')
                ax[i].set_xlabel(x_labels[i])
            ax[1].set_title(key)
            fig.tight_layout()
            plt.savefig(os.path.join('log/images', '%s.png' % key), dpi=400)
            plt.close()

        for k1 in aggregate_results:
            for k2 in aggregate_results[k1]:
                if len(aggregate_results[k1][k2]) > 0:
                    aggregate_results[k1][k2] = [
                        multi_to_single([e[0] for e in aggregate_results[k1][k2]], tag='metric'),
                        multi_to_single([e[1] for e in aggregate_results[k1][k2]], tag='metric'),
                        multi_to_single([e[2] for e in aggregate_results[k1][k2]], tag='metric'),
                        multi_to_single([e[3] for e in aggregate_results[k1][k2]], tag='acc'),
                    ]

        if len(aggregate_results) == 0:
            print('Not data to plot')
            return None

        for key, result_list in aggregate_results.items():
            plot_one_image(key, result_list)

    def parse_dict_keys(self, config, front=''):
        dict_keys = []
        for key in config:
            if isinstance(config[key], dict):
                if len(front) == 0:
                    dict_keys += self.parse_dict_keys(config[key], key)
                else:
                    dict_keys += self.parse_dict_keys(config[key], front + '$$' + key)
            else:
                if len(front) == 0:
                    dict_keys.append(key)
                else:
                    dict_keys.append(front + '$$' + key)
        return dict_keys

    def recursive_retrieve(self, dict_data, string_keys):
        string_keys = string_keys.split('$$')
        for i in range(len(string_keys)):
            key = string_keys[i]
            if key not in dict_data:
                return None
            if isinstance(dict_data[key], dict) and i < (len(string_keys) - 1):
                return self.recursive_retrieve(dict_data[key], '$$'.join(string_keys[1:]))
            else:
                return dict_data[key]

    def retrieve_diff_configs(self):
        results = []
        for i in range(len(self.configs)):
            results.append([
                self.recursive_retrieve(self.configs[i], e)
                for e in self.diff_keys if e not in self.omit_keys
            ])
        return results

    def parse_results(self):
        results = []
        for i in range(len(self.results)):
            self.csv_result_keys[1][0] = 'best_metric$$test_%s' % \
                                         self.configs[i]['model_config']['MLModel']['metrics'][0]
            tmp = []
            for key, process_func in self.csv_result_keys:
                tmp += process_func(self.recursive_retrieve(self.results[i], key))
            results.append(tmp)
        return results

    def aggregate_csv_results(self):
        import numpy as np
        average_results = {}
        for i in range(len(self.configs_diff)):
            key = '$$'.join([str(e) for e in self.configs_diff[i]])
            if key not in average_results:
                average_results[key] = []
            average_results[key].append(self.csv_results[i])
        results = [['Repeat'] + [e.split('$$')[-1] for e in self.diff_keys if e not in self.omit_keys]
                   + [e[0] for e in self.csv_result_keys]]
        for key in average_results:
            average = []
            std = []
            for k in range(len(average_results[key][0])):
                tmp = []
                for j in range(len(average_results[key])):
                    if average_results[key][j][k] is not None:
                        tmp.append(average_results[key][j][k])
                if len(tmp) > 0:
                    average.append('%.5f' % np.mean(tmp))
                    std.append('%.5f' % np.std(tmp))
                else:
                    average.append('NA')
                    std.append('NA')
            results.append(
                [len(average_results[key])] + key.split('$$') +
                ['%s(%s)' % (average[i], std[i]) for i in range(len(average))]
            )
        return results

    def to_csv(self, file_name='average_results.csv'):
        with open(file_name, 'w') as f:
            for e in self.average_results:
                f.write(', '.join([str(e1) for e1 in e]) + '\n')

    @staticmethod
    def process_central_simulate_results(log_files):
        result_dict = {}
        for log_file in log_files:

            with open(log_file, 'r') as f:
                central_simulation = f.readlines()

            if not central_simulation[-1].startswith('Best TEST Metric'):
                print('Incomplete log file, Skip')
                os.remove(log_file)
                continue

            duration = central_simulation[-3].strip('\n').split(' ')[-1]
            val_acc = central_simulation[-2].strip('\n').split(', ')[-1]
            test_acc = central_simulation[-1].strip('\n').split(', ')[-1]
            num_rounds = central_simulation[-5].split(',')[0]

            result_dict[central_simulation[-4]] = result_dict.get(central_simulation[-4], []) + [[
                int(num_rounds), float(duration), float(val_acc), float(test_acc)]]

            print(central_simulation[-4].strip('\n'), val_acc, test_acc, 'max round',
                  central_simulation[-5].split(',')[0])

        results = []
        for key in result_dict:
            acc_mean = ', '.join(np.mean(result_dict[key], axis=0).astype(str).tolist())
            acc_std = ', '.join(np.std(result_dict[key], axis=0).astype(str).tolist())
            results.append(
                str(len(result_dict[key])) + ', ' + key.strip('\n') + ', ' + acc_mean + ', ' + acc_std + '\n'
            )
            print('Repeat', key.strip('\n'), len(result_dict[key]))

        with open('simulate_central.csv', 'w') as f:
            f.write('Repeat, Dataset, #Clients, LR, '
                    'Round, Duration, ValAcc, TestAcc, '
                    'RoundStd, DurationStd, ValStd, TestStd\n')
            f.writelines(results)

    @staticmethod
    def process_fedsgd_simulate_results(log_files):
        result_dict = {}
        for log_file in log_files:

            with open(log_file, 'r') as f:
                fed_sgd_simulation = f.readlines()

            if not fed_sgd_simulation[-1].startswith('Best Metric'):
                print('Incomplete log file, Skip')
                os.remove(log_file)
                continue

            test_acc = fed_sgd_simulation[-1].strip('\n').split(', ')[-1]

            result_dict[fed_sgd_simulation[0]] = result_dict.get(fed_sgd_simulation[0], []) + [[
                int(fed_sgd_simulation[-2].split(',')[0]), float(test_acc)]]

            print(fed_sgd_simulation[0].strip('\n'), test_acc, 'max round', fed_sgd_simulation[-2].split(',')[0])

        results = []
        for key in result_dict:
            acc_mean = ', '.join(np.mean(result_dict[key], axis=0).astype(str).tolist())
            acc_std = ', '.join(np.std(result_dict[key], axis=0).astype(str).tolist())
            results.append(
                str(len(result_dict[key])) + ', ' + key.strip('\n') + ', ' + acc_mean + ', ' + acc_std + '\n'
            )
            print('Repeat', key.strip('\n'), len(result_dict[key]))

        with open('simulate_fed_sgd.csv', 'w') as f:
            f.write('Repeat, Dataset, #Clients, LR, Round, TestAcc, RoundStd, TestStd\n')
            f.writelines(results)

    @staticmethod
    def process_local_simulate_results(log_files):
        result_dict = {}
        for log_file in log_files:

            with open(log_file, 'r') as f:
                local_simulation = f.readlines()

            if not local_simulation[-1].startswith('Average Best Test Metric'):
                print('Incomplete log file, Skip')
                os.remove(log_file)
                continue

            test_acc = local_simulation[-1].strip('\n').split(', ')[-1]

            result_dict[local_simulation[-2]] = result_dict.get(local_simulation[-2], []) + [[float(test_acc)]]

            print(local_simulation[-2].strip('\n'), test_acc)

        results = []
        for key in result_dict:
            acc_mean = ', '.join(np.mean(result_dict[key], axis=0).astype(str).tolist())
            acc_std = ', '.join(np.std(result_dict[key], axis=0).astype(str).tolist())
            results.append(
                str(len(result_dict[key])) + ', ' + key.strip('\n') + ', ' + acc_mean + ', ' + acc_std + '\n'
            )
            print('Repeat', key.strip('\n'), len(result_dict[key]))

        with open('simulate_local.csv', 'w') as f:
            f.write('Repeat, Dataset, #Clients, LR, TestAcc, TestStd\n')
            f.writelines(results)