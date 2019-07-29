import scipy.optimize as opt
from scipy.stats import linregress
from types import FunctionType
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .transform import DEFAULT_TRANSFORM
from .utils import *


def r_squared(x, y):
    """ Return R^2 where x and y are array-like."""
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value ** 2


def corr(x, y):
	""" Return the correlation coefficient between x and y """
	return abs(np.corrcoef(x, y)[0][1])


_METRICS = {
	'corr': corr,
	'r2': r_squared
}


def find_best_transformation(x, y, transformations=None, metric='corr', 
							ignore_na=True, suppress_warning=True):
	"""" Find the best transformation for `x` to linearize the relationship between `x` and `y` 
	:param transformations: A list of Transformer object.
	:param metric: The metric to maximize, default using correlation coefficient.
	:param ignore_na: Whether to ignore nan, default set as True.
	:param warning: Whether to suppress warnings during the fit process
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

	# a
	with warnings.catch_warnings():
		action = 'ignore' if suppress_warning else 'always'
		warnings.simplefilter(action)

		result = []
		for trf in trfs:
			if not trf.validate_input(x):
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


class Linearizer(BaseEstimator, TransformerMixin):

	def __init__(self, cols=None, bins=50, 
				transformations=None, metric='corr', ignore_na=True, copy=True):
		self.cols = cols
		self.bins = 50
		self.cand_trfs = transformations
		self.metric = metric
		self.ignore_na = ignore_na
		self.copy = copy

	def fit(self, X, y):
		cols = self.cols or X.columns

		self.transformations = {}
		for col in cols:
			_, trf = find_best_transformation(X[col], y, 
										      transformations=self.cand_trfs, 
											  metric=self.metric,
											  ignore_na=self.ignore_na)
			self.transformations[col] = trf
		return self

	def transform(self, X):
		check_is_fitted(self, 'transformations')
		
		if self.copy:
			X = X.copy()

		for col, trf in self.transformations:
			X[col] = trf.transform(X[col])

		return X