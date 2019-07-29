import inspect
import functools
import numpy as np
import math


class BaseTransformer:
        
    def __init__(self):
        self.params = None
        
    def __call__(self, x):
        return NotImplementedError
    
    def get_params(self):
        """ Return the variable name for the parameters """
        sig = inspect.signature(self.__call__)
        params = [k for k in sig.parameters if k != 'x']
        return params
    
    def set_params(self, params=None, **kwargs):
        """ Accept both passing parameters as a dictionary and keyword arguments"""
        self.params = params if params is not None else kwargs
        
    def transform(self, x):
        if self.params is None:
            raise ValueError('No parameter values, call the set_params method first with the fitted parameter value.')
        fn = functools.partial(self.__call__, **self.params)
        return fn(x)


class Abs(BaseTransformer):

    def __call__(self, x, a):
        return np.abs(x + a)


class Loge(BaseTransformer):

    def __call__(self, x, a):
        return np.log(x + a)


 class log2(BaseTransformer):

    def __call__(self, x, a):
        return np.log2(x + a)


class log10(BaseTransformer):

    def __call__(self, x, a):
        return np.log10(x + a)


class Exp(BaseTransformer):

    def __call__(self, x, a):
        return np.exp(x + a)


class _Power(BaseTransformer):
    n = 1

    def __call__(self, x, a):
        if self.n > 0:
            return np.power(x + a, self.n)
        else:
            return 1 / (np.power(x + a), self.n) + 1e-15


class Power2(_Power):
    n = 2


class Power3(_Power):
    n = 3


class Power4(_Power):
    n = 4


class Sqrt(_Power):
    n = 1 / 2


class Inv(_Power):
    n = -1


class InvPower2(_Power):
    n = -2
