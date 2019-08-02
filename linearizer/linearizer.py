import scipy.optimize as opt
from scipy.stats import linregress
from types import FunctionType
import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .transform import DEFAULT_TRANSFORM
from .utils import *


__all__ = ['find_best_transformation', 'Linearizer']


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


def find_best_transformation(x, y, transformations=None,
							metric='corr', min_delta=0.0, 
							suppress_warning=True):
	"""" Find the best transformation for `x` to linearize the relationship between `x` and `y` 
	:param transformations: A list of Transformer object.
	:param metric: The metric to maximize, default using correlation coefficient
	:param min_delta: Minimum improvement in the metrics after the transformation
	:param ignore_na: Whether to ignore nan, default set as True
	:param suppress_warning: Whether to suppress warnings during the fit process
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

	# the baseline is the metric under no transformation
	baseline = metric(x, y)

	with warnings.catch_warnings():
		action = 'ignore' if suppress_warning else 'always'
		warnings.simplefilter(action)

		result = []
		for trf in trfs:
			# instantiate a new object
			trf = trf()

			if not trf.validate_input(x):
				continue

			# fit the Transformer
			params = trf.get_params()
			
			try:
				estimation, _ = opt.curve_fit(trf, x, y)
			except RuntimeError:
				# current transformation has a terrible fit, thus ignore it
				continue

			trf.set_params(dict(zip(params, estimation)))

			# the metric after transformation
			m = metric(trf.transform(x), y)
			if m > baseline + min_delta:
				result.append((m, trf))

	if result:
		return sorted(result, reverse=True)[0]
	else:
		# no transformation is needed
		return None, None


class Linearizer(BaseEstimator, TransformerMixin):

	def __init__(self, cols=None, binary_label=True, bins=30, transform_y=None,
				transformations=None, metric='corr', min_delta=0.2,
				ignore_na=True, suppress_warning=True,
				copy=True):
		"""
		:param cols: Choose columns to apply transformations, set as None for all columns.
		:param binary_label: Whether the label is binary (0, 1), in other words. whether the problem
					is classification or regression.
		:param bins: Number of bins when we need to calculate the positive rate in each bins,
					only used when `binary_label` is True.
		:param transform_y: Transformation applied to y, can either be a string within ['odds', 'logodds'], 
                    or a function
		:param transformations: A list of `Transformer` object as the candidate transformations.
		:param metric: The metric to maximize, default using correlation coefficient.
		:param min_delta: Minimum improvement in the metrics after the transformation.
		:param ignore_na: Whether to ignore nan, default set as True.
		:param suppress_warning: Whether to suppress warnings during the fit process
		"""
		self.cols = cols
		self.bins = bins
		self.binary_label = binary_label
		self.transform_y = transform_y
		self.cand_trfs = transformations
		self.metric = metric
		self.ignore_na = ignore_na
		self.suppress_warning = suppress_warning
		self.copy = copy

	def fit(self, X, y):
		cols = self.cols or X.columns

		self.transformations = {}
		for col in cols:
			x_, y_ = preprocess(X[col], y, 
								binary_label=self.binary_label, 
								bins=self.bins, 
								transform_y=self.transform_y, 
								interval_value='mean', 
								ignore_na=self.ignore_na)
			
			_, trf = find_best_transformation(x_, y_, 
										      transformations=self.cand_trfs, 
											  metric=self.metric,
											  suppress_warning=self.suppress_warning)
			self.transformations[col] = trf
		
		return self

	def transform(self, X):
		check_is_fitted(self, 'transformations')
		
		if self.copy:
			X = X.copy()

		for col, trf in self.transformations.items():
			if trf is not None:
				X[col] = trf.transform(X[col])

		return X