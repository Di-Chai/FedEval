import threading
from typing import Dict


class Singleton(object):
    """the base class of singletons.
    Each cls on the inheritance tree can own only one instance.
    """
    _instance_lock = threading.Lock()

    @classmethod
    def _already_got_one(cls):
        return cls in Singleton._instance_dict

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance_dict"):
            Singleton._instance_dict: Dict[type, object] = dict()

        if not cls._already_got_one():
            with Singleton._instance_lock:
                if not cls._already_got_one():
                    _instance = super().__new__(cls)
                    Singleton._instance_dict[cls] = _instance
        return Singleton._instance_dict[cls]
