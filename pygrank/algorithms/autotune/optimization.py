from math import log


def __add(weights, index, increment, max_val, min_val):
    """
    Adds to a value to a specific index in a list of weights while thresholding the outcome to a minimum and maximum value.
    Creates new list of weights holding the result without altering the original.

    Args:
        weights: The list of weights.
        index: The element to add to.
        increment: The value to add.
        max_val: The maximum threshold.
        min_val: The mimimum threshold.
    """
    weights = [weight for weight in weights]
    weights[index] = min(max_val, max(min_val, weights[index]+increment))
    #if increment != 0 and (weights[index] == min_val or weights[index] == max_val):
    #    return None
    # normalize weights
    """
    min_weight = min(0, min(weights))
    if min_weight!=0:
        weights = [weight-min_weight for weight in weights]
    sum_weights = sum(weights)
    weights = [weight/sum_weights for weight in weights]
    """
    return weights


def optimize(loss,
             max_vals=[1 for _ in range(1)],
             min_vals=None,
             deviation_tol: float = 1.E-8,
             divide_range=1.01,
             partitions = 5,
             parameter_tol: float = float('inf'),
             depth=1,
             weights=None,
             verbose=False):
    """
    Implements a coordinate descent algorithm for optimizing the argument vector of the given loss function.
    Arguments:
        loss: The loss function. Could be an expression of the form `lambda p: f(p)' where f takes a list as an argument.
        max_vals: Optional. The maximum value for each parameter to search for. Helps determine the number of parameters.
            Default is a list of ones for one parameter.
        min_vals: Optional. The minimum value for each paramter to search for. If None (default) it becomes a list of
            zeros and equal length to max_vals.
        deviation_tol: Optional. The numerical tolerance of the loss to optimize to. Default is 1.E-8.
        divide_range: Optional. Value greater than 1 with which to divide the range at each iteration. Default is 1.01,
            which guarantees convergence even for difficult-to-optimize functions, but values such as 1.1, 1.2 or 2 may
            also be used for much faster, albeit a little coarser, convergence. Can use "shrinking" to perform s
            hrinking block coordinate descent, but this does not guarantee convergence for some node ranking settings.
        parameter_tol: Optional. The numerical tolerance of parameter values to optimize to. **Both** this and
            deviation_tol need to be met. Default is infinity.
        depth: Optional. Declares the number of times to re-perform the optimization given the previous found solution.
            Default is 1, which only runs the optimization once. Larger depth values can help offset coarseness
            introduced by divide_range.
        weights: Optional. An estimation of parameters to start optimization from. The algorithm tries to center
            solution search around these - hence the usefulness of *depth* as an iterative scheme. If None (default),
            the center of the search range (max_vals+min_vals)/2 is used as a starting estimation.
        verbose: Options. If True, optimization outputs its intermediate steps. Default is False.
    Example:
        >>> import pygrank as pg
        >>> p = pg.optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2, max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5])
        >>> # desired optimization point for the Beale function of this example is [3, 0.5]
        >>> print(p)
        [3.000000052836577, 0.5000000141895036]
    """
    if min_vals is None:
        min_vals = [0 for _ in max_vals]
    #if divide_range<=1:
    #    raise Exception("Need to have a divide_range parameter greater than 1 to actually reduce the search area")
    for min_val, max_val in zip(min_vals, max_vals):
        if min_val > max_val:
            raise Exception("Empty parameter range ["+str(min_val)+","+str(max_val)+"]")
    if str(divide_range)!="shrinking" and divide_range <= 1:
        raise Exception("divide_range should be greater than 1, otherwise the search space never shrinks.")
    #weights = [1./dims for i in range(dims)]
    if weights is None:
        weights = [(min_val+max_val)/2 for min_val, max_val in zip(min_vals, max_vals)]
    range_search = [(max_val-min_val)/2 for min_val, max_val in zip(min_vals, max_vals)]
    curr_variable = 0
    if verbose:
        print("first loss", loss(weights))
    iter = 0
    range_deviations = [float('inf')]*len(max_vals)
    while True:
        assert max(range_search) != 0, "Something went wrong and took too many iterations for optimizer to run (check for nans)"
        if range_search[curr_variable] == 0:
            range_deviations[curr_variable] = 0
            curr_variable += 1
            if curr_variable >= len(max_vals):
                curr_variable -= len(max_vals)
            range_deviations[curr_variable] = 0
            continue
        candidate_weights = [__add(weights, curr_variable, range_search[curr_variable]*(part*2./(partitions-1)-1), max_vals[curr_variable], min_vals[curr_variable]) for part in range(partitions)]
        #print(candidate_weights)
        loss_pairs = [(w,loss(w)) for w in candidate_weights if w is not None]
        #print(loss_pairs)
        weights, _ = min(loss_pairs, key=lambda pair: pair[1])
        if divide_range == "shrinking":
            range_search[curr_variable] = (max_vals[curr_variable]-min_vals[curr_variable])/((iter+1)**0.5*log(iter+2))
        else:
            range_search[curr_variable] /= divide_range
        range_deviations[curr_variable] = max([loss for _, loss in loss_pairs])-min([loss for _, loss in loss_pairs])
        if verbose:
            print('Params', weights, '\t Loss', loss(weights), '+-', max(range_deviations), '\t Var',curr_variable, '\t Parameter max range', max(range_search))
        if max(range_deviations) < deviation_tol and max(range_search) < parameter_tol:
            break
        # move to next var
        iter += 1
        curr_variable += 1
        if curr_variable >= len(max_vals):
            curr_variable -= len(max_vals)
    #print("trained weights in", iter, "iterations", weights, "final loss", loss(weights))
    if depth > 1:
        return optimize(loss, max_vals, min_vals, deviation_tol, divide_range, partitions, parameter_tol, depth - 1, weights, verbose)
    return weights