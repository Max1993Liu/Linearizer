import numpy as np


def drop_na(x, y, according='x'):
	""" Drop the values in both x and y if the element in `according` is missing
		ex. drop_na([1, 2, np.nan], [1, 2, 3], 'x') => [1, 2], [1, 2]
	"""
	if according == 'x':
		valid_index = ~np.isnan(x)
	else:
		valid_index = ~np.isnan(y)

	return np.array(x)[valid_index], np.array(y)[valid_index]

