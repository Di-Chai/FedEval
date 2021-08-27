import json
import os
import numpy as np
import matplotlib.pyplot as plt

from FedEval.run_util import load_config

if __name__ == '__main__':

    # log_path = 'log/ieee_sp_duplicate/Server'
    log_path = 'log/nips_summarize'

    log_path_to_trials = [os.path.join(log_path, e) for e in os.listdir(log_path) if not e.startswith('.')]

    trial_results = []

    for log in log_path_to_trials:
        try:
            c1, c2, c3 = load_config(log)
            with open(os.path.join(log, 'results.json'), 'r') as f:
                results = json.load(f)
            trial_results.append([c1, c2, c3, results])
        except FileNotFoundError as e:
            print('Skip', log, e)

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

        val_loss_list = [
            record[3]['info_each_round'][str(e + 1)]['val_loss'] for e in range(len(record[3]['info_each_round']))
        ]
        best_index = val_loss_list.index(min(val_loss_list))

        # TODO : Del
        # best_index = len(val_loss_list)

        if best_index == 0:
            continue

        test_acc_list = [
            record[3]['info_each_round'][str(e+1)][test_acc_key] for e in range(len(record[3]['info_each_round']))
        ]
        test_acc_list = test_acc_list[:best_index]
        
        # if non_iid == '1':
        #     print(dataset, fl_model, record[3]['best_metric'][test_acc_key], np.mean(record[3]['best_metric_full'][test_acc_key]))

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
        time_to_acc = [
            [record[3]['info_each_round'][str(e+1)]['timestamp'] - record[3]['info_each_round']['1']['timestamp']
             for e in range(1, len(record[3]['info_each_round']))][:best_index],
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
                if i != max_length_index:
                    multi_data[i][0] = multi_data[i][0] + \
                                       multi_data[max_length_index][0][-(max_length-len(multi_data[i][0])):]
                    multi_data[i][1] = multi_data[i][1] + [multi_data[i][1][-1]] * (max_length - len(multi_data[i][1]))
            except:
                print('debug 1')
        print(np.array(multi_data).shape)
        single_data = np.mean(multi_data, axis=0)
        return single_data
    
    datasets = ['mnist', 'femnist', 'celeba', 'semantic140']
    metrics = [cr_to_acc_list, ca_to_acc_list, time_to_acc_list]
    titles = ['CR To Accuracy', 'CA To Accuracy', 'Time To Accuracy']
    x_labels = ['Communication Rounds', 'Communication Amounts (MB)', 'Time (Seconds)']

    fig, ax = plt.subplots(len(metrics), len(datasets), figsize=[4*len(datasets), 4*len(metrics)])

    time_plots = []
    for i in range(len(metrics)):
        for data_id in metrics[i]:
            dataset, non_iid, fl_model = data_id.split()
            print(dataset, non_iid, fl_model)
            if non_iid == '0':
                plot_data = multi_to_single(metrics[i][data_id])
                ax[i][datasets.index(dataset)].plot(
                    plot_data[0], plot_data[1],
                    # label=(fl_model + '-NonIID') if non_iid == '1' else (fl_model + '-IID'),
                    label=fl_model,
                    # linewidth=2.0,
                    alpha=0.8
                )

    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].legend()
            ax[i][j].grid()
            ax[i][j].set_ylabel('Accuracy')
            ax[i][j].set_xlabel(x_labels[i])
            ax[i][j].set_title(' '.join([datasets[j].upper(), titles[i]]))

    # ax.set_ylim(0, 1)
    # ax[0].set_xlabel('Time')
    # ax.set_xticks()
    # ax.set_xticklabels(['10k', '20k', '30k', '40k', '50k'])
    # ax[0].set_ylabel('Accuracy')
    # ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=6))
    
    fig.tight_layout()
    plt.savefig(os.path.join('log/images', 'to_acc_lines.png'), type="png", dpi=400)
    plt.show()
