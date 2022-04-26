import json
import logging
import os
import re
import time
import hashlib
from typing import Any, Mapping

from ..communicaiton import (server_best_weight_filename,
                             weights_filename_pattern)
from ..config import ConfigurationManager
from ..utils.utils import obj_to_pickle_string
from ..strategy.FederatedStrategy import HostParamsType


class HyperLogger:
    _LOG_LEVEL_EVAL_PATTERN = 'logging.{}'

    def __init__(self, name: str, log_dir_name: str) -> None:
        self._name = name

        config_manager = ConfigurationManager()
        rt_cfg = config_manager.runtime_config

        logger = logging.getLogger(name)
        lvl = eval(HyperLogger._LOG_LEVEL_EVAL_PATTERN.format(rt_cfg.base_log_level))
        logger.setLevel(lvl)

        _log_dir_path = os.path.join(config_manager.log_dir_path, log_dir_name)
        self._log_dir_path = os.path.abspath(_log_dir_path)
        os.makedirs(self._log_dir_path, exist_ok=True)
        log_file_path = os.path.join(self._log_dir_path, 'train.log')
        fh = logging.FileHandler(log_file_path, encoding='utf8')
        lvl = eval(HyperLogger._LOG_LEVEL_EVAL_PATTERN.format(rt_cfg.file_log_level))
        fh.setLevel(lvl)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        lvl = eval(HyperLogger._LOG_LEVEL_EVAL_PATTERN.format(rt_cfg.console_log_level))
        ch.setLevel(lvl)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dir_path(self) -> str:
        return self._log_dir_path

    def get(self) -> logging.Logger:
        return logging.getLogger(self._name)

    def snapshot_config_into_files(self) -> None:
        ConfigurationManager().to_files(self._log_dir_path)

    def snapshot_results_into_file(self, results: Mapping[str, Any]) -> None:
        with open(os.path.join(self._log_dir_path, 'results.json'), 'w') as f:
            json.dump(results, f)

    def snapshot_model_weights_into_file(self, weights, round_num: int, host_params_type: str) -> None:
        if weights is None:
            return None
        if host_params_type == HostParamsType.Uniform:
            obj_to_pickle_string(
                weights,
                self.model_weight_file_path(round_num)
            )
            self.clear_snapshot(round_num=round_num, latest_k=1)
        elif host_params_type == HostParamsType.Personalized:
            for client_id in weights:
                client_model_path = self.model_weight_file_path(round_num, client_id=client_id)
                if not os.path.isdir(os.path.dirname(client_model_path)):
                    os.makedirs(os.path.dirname(client_model_path), exist_ok=True)
                obj_to_pickle_string(
                    weights[client_id],
                    self.model_weight_file_path(round_num, client_id=client_id)
                )
            self.clear_snapshot(round_num=round_num, latest_k=1, client_id_list=list(weights.keys()))
        else:
            raise NotImplementedError

    def clear_snapshot(self, round_num: int, client_id_list: list = None, latest_k: int = 1):
        def only_keep_k_model_files(path, k):
            # Keep the latest k weights
            matched_model_files = [
                [re.match(r'model_([0-9]+).pkl', e), e] for e in os.listdir(path)
                if re.match(r'model_([0-9]+).pkl', e)
            ]
            for match, file in matched_model_files:
                if round_num - int(match.group(1)) >= k:
                    try:
                        os.remove(os.path.join(path, file))
                    except FileNotFoundError:
                        # Model does not exist, skip
                        pass
        if client_id_list is None:
            only_keep_k_model_files(
                os.path.dirname(self.model_weight_file_path(round_num=round_num)), k=latest_k
            )
        else:
            for client_id in client_id_list:
                client_log_dir = os.path.dirname(self.model_weight_file_path(round_num=round_num, client_id=client_id))
                if not os.path.isdir(client_log_dir):
                    continue
                only_keep_k_model_files(
                    os.path.dirname(self.model_weight_file_path(round_num=round_num, client_id=client_id)),
                    k=latest_k
                )
                if latest_k <= 0:
                    os.rmdir(client_log_dir)

    def is_snapshot_exist(self, round_num: int, host_params_type: str, client_id_list: list = ()):
        if host_params_type == HostParamsType.Uniform:
            if os.path.isfile(self.model_weight_file_path(round_num)):
                return True
            else:
                return False
        elif host_params_type == HostParamsType.Personalized:
            assert len(client_id_list) > 0
            for client_id in client_id_list:
                client_model_path = self.model_weight_file_path(round_num, client_id=client_id)
                if not os.path.isfile(client_model_path):
                    return False
            return True

    def snap_server_side_best_model_weights_into_file(self, weights) -> None:
        obj_to_pickle_string(weights, file_path=self.server_side_best_model_weight_file_path)

    @property
    def server_side_best_model_weight_file_path(self) -> str:
        return os.path.join(self.dir_path, server_best_weight_filename)

    def model_weight_file_path(self, round_num: int, client_id=None) -> str:
        if client_id is None:
            return os.path.join(self.dir_path, weights_filename_pattern.format(round_num))
        else:
            return os.path.join(self.dir_path, str(client_id), weights_filename_pattern.format(round_num))

    @property
    def log_dir_path(self):
        return self._log_dir_path
