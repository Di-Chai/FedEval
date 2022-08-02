import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

from FedEval.config import ConfigurationManager


if __name__ == '__main__':

    # log_path = 'log/ieee_sp_duplicate/Server'
    # log_path = 'log/JMLR_SGD_Correct'
    log_path = 'log/JMLRCorrect'

    # Debug log
    # LogAnalysis(log_path).to_csv('_'.join(log_path.split('/')) + '.csv')

    log_path_to_trials = [os.path.join(log_path, e, 'Server') for e in os.listdir(log_path)]
    log_path_to_trials = [e for e in log_path_to_trials if os.path.isdir(e)]

    trial_results = []

    for log in log_path_to_trials:
        try:
            c1, c2, c3 = ConfigurationManager.load_configs(log)
            with open(os.path.join(log, 'results.json'), 'r') as f:
                results = json.load(f)
            assert not c3['communication']['limit_network_resource']
            assert c3['docker']['num_containers'] == 8
            assert c3['docker']['num_gpu'] == 0
            trial_results.append([c1, c2, c3, results])
        except FileNotFoundError as e:
            print('Skip', log, e)
        except AssertionError:
            continue

    cr_to_acc_list = {}
    ca_to_acc_list = {}
    time_to_acc_list = {}
    for record in trial_results:

        dataset = record[0]['dataset']
        non_iid = '1' if record[0]['non-iid'] else '0'
        fl_model = record[1]['FedModel']['name']

        if dataset == 'semantic140':
            test_acc_key = 'test_binary_accuracy'
        else:
            test_acc_key = 'test_accuracy'

        test_loss_list = [
            record[3]['info_each_round'][str(e + 1)].get('test_loss') for e in range(len(record[3]['info_each_round']))
        ]
        best_index = test_loss_list.index(min(test_loss_list))

        if best_index == 0:
            continue

        test_acc_list = [
            record[3]['info_each_round'][str(e+1)][test_acc_key] for e in range(len(record[3]['info_each_round']))
        ]
        test_acc_list = test_acc_list[:best_index]

        # CommRound to Accuracy
        cr_to_acc = [
            [e + 1 for e in range(len(record[3]['info_each_round']))][:best_index], test_acc_list
        ]
        # CommAmount to Accuracy
        ca_avg_round = (record[3]['server_send'] + record[3]['server_receive']) / len(record[3]['info_each_round'])
        ca_avg_round_client = ca_avg_round / record[2]['server']['num_clients'] * 2**10  # MB
        ca_to_acc = [
            [(e+1) * ca_avg_round_client for e in range(len(record[3]['info_each_round']))][:best_index],
            test_acc_list
        ]
        # Time to Accuracy
        # time_to_acc = [
        #     [record[3]['info_each_round'][str(e+1)]['timestamp'] - record[3]['info_each_round']['1']['timestamp']
        #      for e in range(1, len(record[3]['info_each_round']))][:best_index],
        #     test_acc_list
        # ]
        federated_time_list = [
            e['max_train'] + e['train_agg'] + e['eval_agg'] + e['max_eval']
            for e in record[3]['federated_time_each_round']
        ]
        time_to_acc = [
            [sum(federated_time_list[:e+1]) for e in range(len(record[3]['federated_time_each_round']))][:best_index],
            test_acc_list
        ]

        data_id = ' '.join([dataset, non_iid, fl_model])

        assert len(cr_to_acc[0]) == len(cr_to_acc[1])
        assert len(ca_to_acc[0]) == len(ca_to_acc[1])
        assert len(time_to_acc[0]) == len(time_to_acc[1])

        cr_to_acc_list[data_id] = cr_to_acc_list.get(data_id, []) + [cr_to_acc]
        ca_to_acc_list[data_id] = ca_to_acc_list.get(data_id, []) + [ca_to_acc]
        time_to_acc_list[data_id] = time_to_acc_list.get(data_id, []) + [time_to_acc]

    def multi_to_single(multi_data):
        max_length = max([len(e[0]) for e in multi_data])
        max_length_index = [len(e[0]) for e in multi_data].index(max_length)
        for i in range(len(multi_data)):
            try:
                if len(multi_data[i][0]) < max_length:
                    multi_data[i][0] = multi_data[i][0] + \
                                       multi_data[max_length_index][0][-(max_length-len(multi_data[i][0])):]
                    multi_data[i][1] = multi_data[i][1] + [multi_data[i][1][-1]] * (max_length - len(multi_data[i][1]))
            except:
                print('debug 1')
        print(np.array(multi_data).shape)
        single_data = np.mean(multi_data, axis=0)
        return single_data
    
    datasets = ['mnist', 'femnist', 'celeba', 'semantic140', 'shakespeare']
    metrics = [cr_to_acc_list, ca_to_acc_list, time_to_acc_list]
    titles = ['CR To Accuracy', 'CA To Accuracy', 'Time To Accuracy']
    x_labels = ['Communication Rounds', 'Communication Amounts (MB)', 'Time (Seconds)']

    # plt.rcParams['font.family'] = 'Times'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    fig, ax = plt.subplots(len(datasets), len(metrics), figsize=[4*len(metrics), 3*len(datasets)])

    color = {
        'FedSGD': 'blue', 'FedAvg': 'orange', 'FedProx': 'purple', 'FedOpt': 'green', 'FedSTC': 'red'
    }

    order = ['FedSGD', 'FedAvg', 'FedProx', 'FedSTC', 'FedOpt']

    time_plots = []
    for i in range(len(metrics)):
        for data_id in sorted(metrics[i].keys(), key=lambda x: order.index(x.split(' ')[-1])):
            dataset, non_iid, fl_model = data_id.split()
            print(dataset, non_iid, fl_model)
            if non_iid == '1':
                plot_data = multi_to_single(metrics[i][data_id])
                ax[datasets.index(dataset)][i].plot(
                    plot_data[0], plot_data[1],
                    # label=(fl_model + '-NonIID') if non_iid == '1' else (fl_model + '-IID'),
                    label=fl_model,
                    # linewidth=2.0,
                    alpha=0.8,
                    # color=color.get(fl_model)
                )

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].legend()
            ax[i][j].grid()
            ax[i][j].set_ylabel('Accuracy')
            ax[i][j].set_xlabel(x_labels[j])
            # ax[i][j].set_title(' '.join([datasets[j].upper(), titles[i]]), fontweight="bold")
            ax[i][j].set_title(' '.join([datasets[i].upper()]), fontweight="bold")
            # if j == 0:
            #     ax[i][j].set_ylim([0.9, 1])
            # elif j == 1:
            #     ax[i][j].set_ylim([0.5, 0.9])
            # elif j == 3:
            #     ax[i][j].set_ylim([0.6, 0.8])

    # ax.set_ylim(0, 1)
    # ax[0].set_xlabel('Time')
    # ax.set_xticks()
    # ax.set_xticklabels(['10k', '20k', '30k', '40k', '50k'])
    # ax[0].set_ylabel('Accuracy')
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=6))
    
    fig.tight_layout()
    plt.savefig(os.path.join('log/images', 'to_acc_lines.pdf'), type="png", dpi=500)
    plt.show()
