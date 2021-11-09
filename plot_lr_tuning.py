from FedEval.run_util import LogAnalysis

log = LogAnalysis('log/tunelr/Server')

log.plot(
    join_keys=['data_config$$dataset', 'model_config$$FedModel$$name'],
    label_keys=['model_config$$MLModel$$optimizer$$lr']
)
