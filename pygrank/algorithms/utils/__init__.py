from pygrank.algorithms.utils.convergence import *
from pygrank.algorithms.utils.optimization import *
from pygrank.algorithms.utils.preprocessing import *
from pygrank.algorithms.utils.graph_signal import *
from pygrank.algorithms.utils.krylov_space import *
import inspect


def _call(method, kwargs):
    """
    This method wraps an argument extraction process and passes only the valid arguments of a given dict to a method.
    This is equivalent to calling method(**kwargs) while ignoring unused arguments.

    Example:
        >>> def func1(arg1):
        >>>     print(arg1)
        >>> def func2(arg2):
        >>>     print(arg2)
        >>> def func(**kwargs):
        >>>     _call(func1, kwargs)
        >>>     _call(func2, kwargs)
        >>> func(arg1="passed to func 1", arg2="passed to func 2")
    """
    return method(**{argname: kwargs[argname] for argname in inspect.signature(method).parameters if argname in kwargs})


def _ensure_all_used(kwargs, methods):
    """
    Makes sure that all named arguments passed to a method reside in the callee methods.

    Example:
        >>> def func(**kwargs):
        >>>     _call(func1, kwargs)
        >>>     _call(func2, kwargs)
        >>>     _ensure_all_used(kwargs, [func1, func2])
    """
    all_args = list()
    for method in methods:
        all_args.extend(inspect.signature(method).parameters.keys())
    missing = set(kwargs.keys())-set(all_args)
    if len(missing) != 0:
        raise Exception("No usage of argument(s) "+str(missing)+" found")