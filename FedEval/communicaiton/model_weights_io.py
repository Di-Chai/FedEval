import requests
import pickle
from abc import ABCMeta, abstractmethod

weights_filename_pattern = 'model_{}.pkl'  # filled with round number
server_best_weight_filename = 'best_model.pkl'


class ModelWeightsIoInterface(metaclass=ABCMeta):
    @abstractmethod
    def fetch_params(self, filename: str):
        raise NotImplementedError


class ModelWeightsHandler(ModelWeightsIoInterface):
    def __init__(self, download_url_pattern: str):
        if '{}' not in download_url_pattern:
            raise ValueError(f"weights_download_url_pattern is not a pattern:" +
                             " '{weights_download_url_pattern}' does not contain " + '{}')
        self._download_url_pattern: str = download_url_pattern
        self._timeout: int = 600

    def fetch_params(self, file_location: str):
        protocol = 'http://'

        def _fetch():
            return requests.get(protocol + self._download_url_pattern.format(
                file_location), timeout=self._timeout)

        exceed_time = 10
        counter = 0
        response = _fetch()
        while response.status_code != 200:
            response = _fetch()
            counter += 1
            assert counter < exceed_time, 'Exceed maximum model download times.'
        return pickle.loads(response.content)
