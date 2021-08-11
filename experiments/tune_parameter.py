from pygrank.algorithms import optimize
from pygrank.algorithms import GenericGraphFilter
from experiments.importer import import_SNAP
from pygrank.algorithms import to_signal, preprocessor
from pygrank.measures.utils import split
from pygrank.measures import AUC

dataset = "ant"
G, groups = import_SNAP(dataset, max_group_number=1)
group = groups[0]

known, evaluation = split(list(group), training_samples=0.2)
training, validation = split(known, training_samples=0.5)
training, validation, evaluation = to_signal(G,{v: 1 for v in training}), to_signal(G,{v: 1 for v in validation}), to_signal(G,{v: 1 for v in evaluation})

pre = preprocessor("symmetric", True)
params = optimize(lambda params: -AUC(evaluation, exclude=known).evaluate(GenericGraphFilter([params[0]]*5+[params[1]]*5, to_scipy=pre, max_iters=10000).rank(G, training)),
                  max_vals=[0.99, 0.99], min_vals=[0.1, 0.1], deviation_tol=0.01, verbose=True, divide_range="shrinking", partitions=5)
print(AUC(evaluation, exclude=known).evaluate(GenericGraphFilter([params[0]]*5+[params[1]]*5, to_scipy=pre, max_iters=10000).rank(G, training)))