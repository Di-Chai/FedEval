import json
import os
import numpy as np

from FedEval.run_util import load_config

if __name__ == '__main__':

    log_path = 'log/ieee_sp_duplicate/Server'

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

    for record in trial_results:
        # CommRound to Accuracy
        cr_to_acc = [
            [e + 1 for e in range(len(record[3]['info_each_round']))],
            [record[3]['info_each_round'][str(e+1)]['test_accuracy']
             for e in range(len(record[3]['info_each_round']))]
        ]
        # CommAmount to Accuracy
        ca_avg_round = (record[3]['server_send'] + record[3]['server_receive']) / len(record[3]['info_each_round'])
        ca_avg_round_client = ca_avg_round / record[2]['server']['num_clients'] * 2**10  # MB
        ca_to_acc = [
            [(e+1) * ca_avg_round_client for e in range(len(record[3]['info_each_round']))],
            [record[3]['info_each_round'][str(e+1)]['test_accuracy']
             for e in range(len(record[3]['info_each_round']))]
        ]
        # Time to Accuracy
        time_to_acc = [
            [0] +
            [record[3]['info_each_round'][str(e+1)]['timestamp'] - record[3]['info_each_round']['1']['timestamp']
             for e in range(1, len(record[3]['info_each_round']))],
            [record[3]['info_each_round'][str(e+1)]['test_accuracy']
             for e in range(len(record[3]['info_each_round']))]
        ]
        time_to_acc[0][0] = \
            np.multiply([int(e) for e in record[3]['total_time'].split(':')], [3600, 60, 1]).sum() - time_to_acc[0][-1]
        time_to_acc[0][0] = max(time_to_acc[0][0], 0)
        print('debug')