from abc import ABC, abstractmethod
from platform import system
from typing import Tuple

import psutil


def _get_comm_in_and_out_linux() -> Tuple[int, int]:
    """retrieve network traffic counter in linux platforms (with the support from psutil lib).

    Returns:
        Tuple[int, int]: (the number of received bytes, the number of sent bytes)
    """
    eth0_info = psutil.net_io_counters(pernic=True).get('eth0')
    bytes_recv = 0 if eth0_info is None else eth0_info.bytes_recv
    bytes_sent = 0 if eth0_info is None else eth0_info.bytes_sent
    return bytes_recv, bytes_sent


class Communicatior(ABC):
    # @abstractmethod
    # def on(self, event, *args, **kwargs) -> None:
    #     pass

    # @abstractmethod
    # def invoke(self, event, *args, **kwargs) -> Any:
    #     pass
    pass

    @staticmethod
    def get_comm_in_and_out() -> Tuple[int, int]:
        """retrieve network traffic counter.

        Raises:
            NotImplementedError: raised when called at unsupported
                platforms or unknown platforms.

        Returns:
            Tuple[int, int]: (the number of received bytes, the number of sent bytes)
        """
        platform_str = system().lower()
        if platform_str == 'linux':
            return _get_comm_in_and_out_linux()
        elif platform_str == 'windows':
            raise NotImplementedError(
                f'Unsupported function at {platform_str} platform.')
        else:
            raise NotImplementedError("Unknown platform.")
