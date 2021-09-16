from functools import wraps
from threading import Lock
from typing import Callable, Dict, Optional


class Singleton(object):
    """the base class of singletons.
    Each cls on the inheritance tree can own only one instance.
    """
    _instance_lock = Lock()

    # to protect all writable attributes in all singleton classes
    _writable_lock: Optional[Lock] = None

    __init_once_lock = Lock()   # thread lock for __initiated
    __initiated = False # whether this class has been initiated

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

    def __init__(self, thread_safe = False) -> None:
        if thread_safe:
            Singleton._writable_lock = Lock()   # can be reset by re-initiate
        with Singleton.__init_once_lock:
            if not Singleton.__initiated:
                super().__init__()
                Singleton.__initiated = True

    @classmethod
    def thread_safe_ensurance(cls, func: Callable):
        @wraps(func)
        def thread_safe_ensurance(*args, **kwargs):
            if cls._writable_lock:
                cls._writable_lock.acquire()
            res = func(*args, **kwargs)
            if cls._writable_lock:
                cls._writable_lock.release()
            return res
        return thread_safe_ensurance

    @staticmethod
    def thread_safe():
        return Singleton._writable_lock is not None
