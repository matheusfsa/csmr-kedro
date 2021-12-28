from importlib import import_module
from sklearn.model_selection import ParameterGrid, ParameterSampler
import scipy.stats as stats
import numpy as np

def model_from_string(model_name, default_args=None):
    model_class = getattr(
        import_module((".").join(model_name.split(".")[:-1])),
        model_name.rsplit(".")[-1],
    )
    if default_args is None:
        return model_class()
    else:
        return model_class(**default_args)


def load_params(params):
    for p in params:
        if isinstance(params[p], str):
            params[p] = eval(params[p])
        elif isinstance(params[p], list):
            params[p] = list(map(lambda x: tuple(x) if isinstance(x, list) else x, params[p]))
    return params