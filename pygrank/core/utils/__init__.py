from pygrank.core.utils.preprocessing import *
import inspect
import sys


def log(text=""):
    sys.stdout.write("\r"+text)
    sys.stdout.flush()


def call(method, kwargs, args=None):
    """
    This method wraps an argument extraction process and passes only the valid arguments of a given dict to a method.
    This is equivalent to calling method(**kwargs) while ignoring unused arguments.

    Example:
        >>> def func1(arg1):
        >>>     print(arg1)
        >>> def func2(arg2):
        >>>     print(arg2)
        >>> def func(**kwargs):
        >>>     call(func1, kwargs)
        >>>     call(func2, kwargs)
        >>> func(arg1="passed to func 1", arg2="passed to func 2")
    """
    if args:
        kwargs = dict(kwargs)
        for arg, val in zip(list(inspect.signature(method).parameters)[:len(args)], args):
            if arg in kwargs:
                raise Exception("Repeated argument to method "+method.__name__+": "+arg)
            kwargs[arg] = val
    return method(**{kwarg: kwargs[kwarg] for kwarg in inspect.signature(method).parameters if kwarg in kwargs})


def remove_used_args(method, kwargs, args=None):
    if args:
        kwargs = dict(kwargs)
        for arg, val in zip(list(inspect.signature(method).parameters)[:len(args)], args):
            kwargs[arg] = val
    params = set(inspect.signature(method).parameters)
    return {kwarg: val for kwarg, val in kwargs.items() if kwarg not in params}


def ensure_used_args(kwargs, methods=None):
    """
    Makes sure that all named arguments passed to a method reside in the callee methods.

    Example:
        >>> def func(**kwargs):
        >>>     call(func1, kwargs)
        >>>     call(func2, kwargs)
        >>>     ensure_used_args(kwargs, [func1, func2])
    """
    all_args = list()
    if methods is not None:
        for method in methods:
            all_args.extend(inspect.signature(method).parameters.keys())
    missing = set(kwargs.keys())-set(all_args)
    if len(missing) != 0:
        raise Exception("No usage of argument(s) "+str(missing)+" found")
