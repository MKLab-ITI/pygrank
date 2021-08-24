import io
import sys


def _fraction2str(num):
    """
    Helper method to pretty print percentages.
    Args:
        num: A number in the range [0,1].
    """
    if isinstance(num, str):
        return num
    if num < 0.005:
        return "0"
    ret = str(int(num*100+.5)/100.)
    if len(ret) < 4:
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


def benchmark_print(benchmark, delimiter: str = " \t ", end_line: str = "", out: object = sys.stdout, err=sys.stderr):
    """
    Print outcomes provided by a given benchmark as a table in the console. To ensure that `sys.stderr`
    does not interrupt printing, this method buffers it and prints all error messages at once in the end.
    (This is made so exception can be traced normally.)

    Args:
        benchmark: A map from names to node ranking algorithms to compare.
        datasets: A list of datasets to compare the algorithms on. List elements should either be strings or (string, num) tuples
            indicating the dataset name and number of community of interest respectively.
        delimiter: How to separate columns. Use " & " when exporting to latex format.
        end_line: What to print before the end of line. Use "\\\\" when exporting to latex format.
        out: The stream in which to print behchmark results. Default is sys.stdout .
        err: The stream in which to print errors. Default is sys.stderr .

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
            print(delimiter.join([_fill(_fraction2str(value), tab) for tab, value in zip(tabs, line)]) + end_line, file=out)
    finally:
        sys.stderr = old_stderr
        out.flush()
        if err is not None:
            print(buffered_error.getvalue(), file=err)
    return out


def benchmark_scores(benchmark):
    return [value for line in benchmark for value in line if not isinstance(value, str)]


def benchmark_dict(benchmark):
    ret = dict()
    names = list()
    for line in benchmark:
        if not names:
            names = line[1:]
        else:
            ret[line[0]] = {name: value for name, value in zip(names, line[1:])}
    return ret
