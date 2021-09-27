import json
import logging
import os
import re
import time
from typing import Any, Mapping

from ..communicaiton import (server_best_weight_filename,
                             weights_filename_pattern)
from ..config import ConfigurationManager
from ..utils.utils import obj_to_pickle_string


class HyperLogger:
    _LOG_LEVEL_EVAL_PATTERN = 'logging.{}'

    def __init__(self, name: str, log_dir_name: str) -> None:
        self._name = name
        rt_cfg = ConfigurationManager().runtime_config
        
        logger = logging.getLogger(name)
        lvl = eval(HyperLogger._LOG_LEVEL_EVAL_PATTERN.format(rt_cfg.base_log_level))
        logger.setLevel(lvl)

        time_str = time.strftime('%Y_%m%d_%H%M%S', time.localtime())
        _log_dir_path = os.path.join(
            rt_cfg.log_dir_path, log_dir_name, time_str)
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

    def snapshot_model_weights_into_file(self, weights, round_num: int) -> None:
        obj_to_pickle_string(
            weights,
            self.model_weight_file_path(round_num)
        )
        # Keep the latest 5 weights
        all_files_in_model_dir = os.listdir(self._log_dir_path)
        matched_model_files = [
            re.match(r'model_([0-9]+).pkl', e) for e in all_files_in_model_dir]
        matched_model_files = [e for e in matched_model_files if e is not None]
        for matched_model in matched_model_files:
            if round_num - int(matched_model.group(1)) >= 5:
                os.remove(os.path.join(self._log_dir_path, matched_model.group(0)))

    def snap_server_side_best_model_weights_into_file(self, weights) -> None:
        obj_to_pickle_string(weights, file_path=self.server_side_best_model_weight_file_path)

    @property
    def server_side_best_model_weight_file_path(self) -> str:
        return os.path.join(self.dir_path, server_best_weight_filename)

    def model_weight_file_path(self, round_num: int) -> str:
        return os.path.join(self.dir_path, weights_filename_pattern.format(round_num))
