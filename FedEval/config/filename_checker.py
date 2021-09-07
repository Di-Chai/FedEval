from functools import wraps
from typing import Callable, List, Optional, Sequence, Union

_PATH_SEPERATORS = ['\\', '/']

def check_filename(idx: Union[int, Sequence[int]]=0, allowed_suffixes: Optional[Sequence[str]]=None):
    def _check_filename(func: Callable):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            filenames: List[str] = [args[idx]] if isinstance(
                idx, int) else [args[i] for i in idx]
            for filename in filenames:
                for sep in _PATH_SEPERATORS:
                    if sep in filename:
                        raise ValueError(
                            f"found path seperator(s) in the given filename({filename}).")
                if allowed_suffixes:
                    # TODO(fgh) add unit tests in test_config.py
                    end_valid = False
                    for a_s in allowed_suffixes:
                        if filename.endswith(a_s):
                            end_valid = True
                            break
                    if not end_valid:
                        raise ValueError(
                            f'{filename} dose not end with an allowed suffix.')
            return func(*args, **kwargs)
        return _wrapper
    return _check_filename
