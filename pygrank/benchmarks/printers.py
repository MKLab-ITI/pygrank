import io
import math
import sys
from typing import Optional
from scipy import stats
import numpy as np


def _fraction2str(num, decimals=2):
    """
    Helper method to pretty print percentages.
    Args:
        num: A number in the range [0,1].
    """
    if isinstance(num, str):
        return num
    if math.isnan(num):
        return "NaN"
    mult = math.pow(10, decimals)
    if num < 0.5/mult:
        return "0"
    ret = str(int(num*mult+.5)/float(mult))
    if len(ret) < decimals+2:
        ret += "0"
    if ret[0] == "0":
        return ret[1:]
    return ret


def _fill(text="", tab=14):
    """
    Helper method to customly align texts by adding trailing spaces up to a fixed point.
    Args:
        text: The text to add spaces to.
        tab: The alignment point. Default is 14.
    Example:
        >>> print(_fill("Text11")+_fill("Text12"))
        >>> print(_fill("Text21")+_fill("Text22"))
    """
    return text+(" "*(tab-len(text)))


def benchmark_print_line(*line,
                    delimiter: str = " \t ",
                    end_line: str = "",
                    decimals: int = 2,
                    out: Optional[io.TextIOWrapper] = sys.stdout,
                    tabs: list = None):
    if tabs is None:
        tabs = [7]*len(line)
        tabs[0] = 14
    print(delimiter.join([_fill(_fraction2str(value, decimals=decimals), tab) for tab, value in zip(tabs, line)]) + end_line, file=out)


def benchmark_print(benchmark,
                    delimiter: str = " \t ",
                    end_line: str = "",
                    out: Optional[io.TextIOWrapper] = sys.stdout,
                    err: Optional[io.TextIOWrapper] = sys.stderr,
                    decimals: int = 2):
    """
    Print outcomes provided by a given benchmark as a table in the console. To ensure that `sys.stderr`
    does not interrupt printing, this method buffers it and prints all error messages at once in the end.
    (This is made so exception can be traced normally.)

    Args:
        benchmark: A mapping from names to node ranking algorithm outcomes to compare.
            Typically this is yielded by benchmarking experiments.
        delimiter: How to separate columns. Use " & " when exporting to latex format.
        end_line: What to print before the end of line. Use "\\\\" when exporting to latex format.
        out: The stream in which to print behchmark results. Default is sys.stdout .
        err: The stream in which to print errors. Default is sys.stderr .
        decimals: How many decimal places to print.
    Example:
        >>> benchmark_print(..., delimiter=" & ", end_line="\\\\") #  latex output
    """
    old_stderr = sys.stderr
    sys.stderr = buffered_error = io.StringIO()
    tabs = None
    try:
        for line in benchmark:
            if tabs is None:
                tabs = [len(value)+1 for value in line]  # first line should be algorithm names
                tabs[0] = 14
            benchmark_print_line(*line, delimiter=delimiter, end_line=end_line, out=out, decimals=decimals, tabs=tabs)
    finally:
        sys.stderr = old_stderr
        out.flush()
        if err is not None:
            print(buffered_error.getvalue(), file=err)
    return out


def benchmark_scores(benchmark):
    return [value for line in benchmark for value in line if not isinstance(value, str)]


def benchmark_ranks(benchmark):
    for line in benchmark:
        numbers = [-value for value in line if not isinstance(value, str)]
        #orders = {i: order+1 for order, i in enumerate(sorted(list(range(len(numbers))), key=lambda i: numbers[i], reverse=True))}
        orders = stats.rankdata(numbers)
        ret = list()
        i = 0
        for value in line:
            if isinstance(value, str):
                ret.append(value)
            else:
                #ret.append(str(value)+" ("+str(orders[i])+")")
                ret.append(orders[i])
                i += 1
        yield ret


def benchmark_average(benchmark, posthocs=False):
    sums = None
    for prev_line in benchmark:
        line = prev_line[1:]
        if not isinstance(line[0], str):
            if sums is None:
                sums = [[value] for value in line]
            else:
                for values, value in zip(sums, line):
                    values.append(value)
        yield prev_line
    if sums is not None:
        yield ["Average"]+[sum(values)/len(values) for values in sums]
    if sums is not None and posthocs:
        import scikit_posthocs as ph
        print(ph.posthoc_nemenyi_friedman(np.array(sums).T))


def benchmark_dict(benchmark):
    ret = dict()
    names = list()
    for line in benchmark:
        if not names:
            names = line[1:]
        else:
            ret[line[0]] = {name: value for name, value in zip(names, line[1:])}
    return ret
