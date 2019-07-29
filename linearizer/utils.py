import numpy as np
import pandas as pd


def drop_na(x, y, according='x'):
    """ Drop the values in both x and y if the element in `according` is missing
        ex. drop_na([1, 2, np.nan], [1, 2, 3], 'x') => [1, 2], [1, 2]
    """
    if according == 'x':
        valid_index = ~np.isnan(x)
    else:
        valid_index = ~np.isnan(y)

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


def as_positive_rate(x, y, bins=30, interval_value='mean'):
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
