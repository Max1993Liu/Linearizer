import numpy as np
import pandas as pd
from types import FunctionType
import warnings

from .transform import BaseTransformer


def drop_na(x, y, according='both'):
    """ Drop the values in both x and y if the element in `according` is missing
        ex. drop_na([1, 2, np.nan], [1, 2, 3], 'x') => [1, 2], [1, 2]
    """
    if according == 'x':
        valid_index = ~np.isnan(x)
    elif according == 'y':
        valid_index = ~np.isnan(y)
    elif according == 'both':
        valid_index = (~np.isnan(x)) & (~np.isnan(y))
    else:
        raise ValueError('According should be one of {}'.format(['x', 'y', 'both']))

    return np.array(x)[valid_index], np.array(y)[valid_index]


def check_binary_label(y):
    """ Make sure the label contains only 0 and 1 """
    if set(y) != set([0, 1]):
        raise ValueError('The label must be binary 0 or 1.')


def check_numerical(x):
    if isinstance(x, list):
        x = x[0]
    if not pd.api.types.is_numeric_dtype(x):
        raise ValueError('The input must be a numerical array.')


def as_positive_rate(x, y, bins, interval_value='mean'):
    """ Group numerical variable x into several bins
        and calculate the positive rate within each bin
    :param bins: Integer or a sequence of values as cutoff points
    :param interval_value: One of ['left', 'right', 'mean'], how the interval is converted to a scalar
    """
    if isinstance(x, list): 
        x = np.array(x)

    check_numerical(x)
    check_binary_label(y)

    if len(set(x)) <= bins:
        pos_pct = pd.Series(y).groupby(x).mean()
    else:
        intervals = pd.cut(x, bins)

        if interval_value == 'left':
            intervals = [i.left for i in intervals]
        elif interval_value == 'right':
            intervals = [i.right for i in intervals]
        elif interval_value == 'mean':
            intervals = [(i.left + i.right) / 2.0 for i in intervals]
        else:
            raise ValueError('Only {} is supported.'.format(['left', 'right', 'mean']))
        
        pos_pct = pd.Series(y).groupby(intervals).mean()
    
    return pos_pct.index.values, pos_pct.values


EPILSON = 1e-15


def _odds(p):
    p = np.clip(p, EPILSON, 1 - EPILSON)
    return p / (1 - p)


def _logodds(p):
    return np.log(_odds(p))


_TRANSFORMS = {
    'odds': _odds,
    'logodds': _logodds
}


def preprocess(x, y, binary_label=True, bins=50, transform_y=None, interval_value='mean', ignore_na=True):
    """ Preprocess the input before finding the best transformations
    :param binary_label: Whether the label is binary (0, 1), in other words. whether the problem
                    is classification or regression.
    :param transform_y: Transformation applied to y, can either be a string within ['odds', 'logodds'], 
                    or a function
    :param bins: Integer or a sequence of values as cutoff points
    :param interval_value: One of ['left', 'right', 'mean'], how the interval is converted to a scalar
    :ignore_na: Whether to ignore NA
    """
    if binary_label:
        x, y = as_positive_rate(x, y, bins, interval_value)

    if transform_y is not None:
        # make sure y is an array
        y = np.array(y)
        
        if isinstance(transform_y, str):
            if transform_y not in _TRANSFORMS:
                raise ValueError('Only {} is supported.'.format(_TRANSFORMS.keys()))
            y = _TRANSFORMS[transform_y](y)
        elif isinstance(transform_y, FunctionType):
            y = transform_y(y)
        else:
            raise ValueError('Only string and function is supported for `transform_y`.')

    if ignore_na:
        x, y = drop_na(x, y, according='both')

    return x, y


def _check_complexity():
    cpl = {}
    for cls in BaseTransformer.__subclasses__():
        complexity = cls.complexity
        if complexity in cpl:
            warnings.warn('{} and {} has the same complexity {}.'.\
                    format(cls.__name__, cpl[complexity].__name__, complexity))
        cpl[complexity] = cls 
