"""
src/neurons_model/ui/utilities.py
Utility functions.
"""


from __future__ import annotations


import os
from collections import defaultdict
from functools import partial
from itertools import repeat
import psutil
import numpy as np


def printmem() -> None:
    """Print current process RSS memory in GB (requires psutil)."""
    if psutil is None:
        print("psutil not available; cannot report memory.")
        return
    
    process = psutil.Process(os.getpid())
    rss_gb = round(process.memory_info().rss / 1e9, 3)
    print(f"Memory usage: {rss_gb} GB")


def nested_defaultdict(default_factory, depth: int = 1):
    """
    Create a nested defaultdict of arbitrary depth.
    """
    if depth < 1:
        raise ValueError("depth must be >= 1")
    
    result = partial(defaultdict, default_factory)
    
    for _ in repeat(None, depth - 1):
        result = partial(defaultdict, result)

    return result()
