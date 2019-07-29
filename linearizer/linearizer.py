import scipy.optimize as opt
from scipy.stats import linregress
import numpy as np
from types import FunctionType

from .transform import DEFAULT_TRANSFORM
from .utils import *


def r_squared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value ** 2


def corr(x, y):
	""" Return the correlation coefficient between x and y """
	return np.corrcoef(x, y)[0][1]


_METRICS = {
	'corr': corr,
	'r2': r_squared
}


def find_best_transformation(x, y, transformations=None, metric='corr', ignore_na=True):
	"""" Find the best transformation for `x` to linearize the relationship between `x` and `y` 
	:param transformations: A list of Transformer object.
	:param metric: The metric to maximize, default using correlation coefficient.
	:param ignore_na: Whether to ignore nan, default set as True.
	"""
	trfs = transformations or DEFAULT_TRANSFORM

	if isinstance(metric, str):
		if metric not in _METRICS:
			raise ValueError('Only supports the following metrics: {}'.format(_METRICS.keys()))
		metric = _METRICS[metric]
	elif isinstance(metric, FunctionType):
		pass
	else:
		raise ValueError('The `metric` argument should either be a string or a function.')

	if ignore_na:
		x, y = drop_na(x, y, according='x')


	# the baseline is the metric under no transformation
	baseline = metric(x, y)

	result = []
	for trf in trfs:
		if not trf.validate(x):
			continue

		# fit the Transformer
		params = trf.get_params()
		estimation, _ = opt.curve_fit(trf, x, y)
		trf.set_params(dict(zip(params, estimation)))

		# the metric after transformation
		m = metric(trf.transform(x), y)
		if m > baseline:
			result.append((m, trf))

	if result:
		return sorted(result, reverse=True)[0]
	else:
		# no transformation is needed
		return None, None
