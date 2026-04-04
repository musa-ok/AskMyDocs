"""Small utilities (e.g. simple timing wrapper for nodes)."""
import time
from functools import wraps


def trace_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        duration = end_time - start_time
        print(f"--- Node: {func.__name__} | Duration: {duration:.4f} seconds ---")

        return result

    return wrapper