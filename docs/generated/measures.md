# Measures
The following measures can be imported from the package `pygrank.measures`.
Constructor details are provided, including arguments inherited from and passed to parent classes.

## <span class="component">Time</span>
<b class="parameters">Extends</b><br> *Measure*<br><b class="parameters">About</b><br>
An abstract class that can be passed to benchmark experiments to indicate that they should report running time 
of algorithms. Instances of this class have no functionality. The constructor initialize self.  See help(type(self)) for accurate signature.
## <span class="component">AM</span>
<b class="parameters">Extends</b><br> *MeasureCombination*<br><b class="parameters">About</b><br>
Combines several measures through their arithmetic mean. The constructor instantiates a combination of several measures. More measures with their own weights and threhsolded range 
can be added with the `add(measure, weight=1, min_val=-inf, max_val=inf)` method. 
<br><b class="parameters">Parameters</b>

 * *measures:* Optional. An iterable of measures to combine. If None (default) no new measure is added. 
 * *weights:* Optional. A iterable of floats with which to weight the measures provided by the previous argument. The concept of weighting depends on how measures are aggregated, but it corresponds to an importance value placed on each measure. If None (default), provided measures are all weighted by 1. 
 * *thresholds:* Optional. A tuple of [min_val, max_val] with which to bound measure outcomes. If None (default) provided measures 
 * *differentiable:* Optional. If True, a differentiable hinge loss is used to approximate max and min. Default is False. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
known_scores, algorithm, personalization, sensitivity_scores = ... 
auc = pg.AUC(known_scores, exclude=personalization) 
prule = pg.pRule(sensitivity_scores, exclude=personalization) 
measure = pg.AM([auc, prule], weights=[1., 10.], thresholds=[(0,1), (0, 0.8)]) 
print(measure(algorithm(personalization))) 
```
Example (same result):
```python 
measure = pg.AM().add(auc, weight=1., max_val=1).add(prule, weight=1., max_val=0.8) 
```
## <span class="component">Disparity</span>
<b class="parameters">Extends</b><br> *MeasureCombination*<br><b class="parameters">About</b><br>
Combines measures by calculating the absolute value of their weighted differences. 
If more than two measures *measures=[M1,M2,M3,M4,...]* are provided this calculates *abs(M1-M2+M3-M4+...)* The constructor instantiates a combination of several measures. More measures with their own weights and threhsolded range 
can be added with the `add(measure, weight=1, min_val=-inf, max_val=inf)` method. 
<br><b class="parameters">Parameters</b>

 * *measures:* Optional. An iterable of measures to combine. If None (default) no new measure is added. 
 * *weights:* Optional. A iterable of floats with which to weight the measures provided by the previous argument. The concept of weighting depends on how measures are aggregated, but it corresponds to an importance value placed on each measure. If None (default), provided measures are all weighted by 1. 
 * *thresholds:* Optional. A tuple of [min_val, max_val] with which to bound measure outcomes. If None (default) provided measures 
 * *differentiable:* Optional. If True, a differentiable hinge loss is used to approximate max and min. Default is False. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
known_scores, algorithm, personalization, sensitivity_scores = ... 
auc = pg.AUC(known_scores, exclude=personalization) 
prule = pg.pRule(sensitivity_scores, exclude=personalization) 
measure = pg.AM([auc, prule], weights=[1., 10.], thresholds=[(0,1), (0, 0.8)]) 
print(measure(algorithm(personalization))) 
```
Example (same result):
```python 
measure = pg.AM().add(auc, weight=1., max_val=1).add(prule, weight=1., max_val=0.8) 
```
## <span class="component">GM</span>
<b class="parameters">Extends</b><br> *MeasureCombination*<br><b class="parameters">About</b><br>
Combines several measures through their geometric mean. The constructor instantiates a combination of several measures. More measures with their own weights and threhsolded range 
can be added with the `add(measure, weight=1, min_val=-inf, max_val=inf)` method. 
<br><b class="parameters">Parameters</b>

 * *measures:* Optional. An iterable of measures to combine. If None (default) no new measure is added. 
 * *weights:* Optional. A iterable of floats with which to weight the measures provided by the previous argument. The concept of weighting depends on how measures are aggregated, but it corresponds to an importance value placed on each measure. If None (default), provided measures are all weighted by 1. 
 * *thresholds:* Optional. A tuple of [min_val, max_val] with which to bound measure outcomes. If None (default) provided measures 
 * *differentiable:* Optional. If True, a differentiable hinge loss is used to approximate max and min. Default is False. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
known_scores, algorithm, personalization, sensitivity_scores = ... 
auc = pg.AUC(known_scores, exclude=personalization) 
prule = pg.pRule(sensitivity_scores, exclude=personalization) 
measure = pg.AM([auc, prule], weights=[1., 10.], thresholds=[(0,1), (0, 0.8)]) 
print(measure(algorithm(personalization))) 
```
Example (same result):
```python 
measure = pg.AM().add(auc, weight=1., max_val=1).add(prule, weight=1., max_val=0.8) 
```
## <span class="component">Parity</span>
<b class="parameters">Extends</b><br> *MeasureCombination*<br><b class="parameters">About</b><br>
Combines measures by calculating the absolute value of their weighted differences subtracted from 1. 
If more than two measures *measures=[M1,M2,M3,M4,...]* are provided this calculates *1-abs(M1-M2+M3-M4+...)* The constructor instantiates a combination of several measures. More measures with their own weights and threhsolded range 
can be added with the `add(measure, weight=1, min_val=-inf, max_val=inf)` method. 
<br><b class="parameters">Parameters</b>

 * *measures:* Optional. An iterable of measures to combine. If None (default) no new measure is added. 
 * *weights:* Optional. A iterable of floats with which to weight the measures provided by the previous argument. The concept of weighting depends on how measures are aggregated, but it corresponds to an importance value placed on each measure. If None (default), provided measures are all weighted by 1. 
 * *thresholds:* Optional. A tuple of [min_val, max_val] with which to bound measure outcomes. If None (default) provided measures 
 * *differentiable:* Optional. If True, a differentiable hinge loss is used to approximate max and min. Default is False. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
known_scores, algorithm, personalization, sensitivity_scores = ... 
auc = pg.AUC(known_scores, exclude=personalization) 
prule = pg.pRule(sensitivity_scores, exclude=personalization) 
measure = pg.AM([auc, prule], weights=[1., 10.], thresholds=[(0,1), (0, 0.8)]) 
print(measure(algorithm(personalization))) 
```
Example (same result):
```python 
measure = pg.AM().add(auc, weight=1., max_val=1).add(prule, weight=1., max_val=0.8) 
```
## <span class="component">AUC</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Wrapper for sklearn.metrics.auc evaluation. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">Accuracy</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the accuracy as 1- mean absolute differences between given and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">BinaryCrossEntropy</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes a cross-entropy loss of given vs known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">Cos</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the cosine similarity between given and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">CrossEntropy</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the KL-divergence of given vs known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">Dot</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the dot similarity between given and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">Euclidean</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the Euclidean distance between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">KLDivergence</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the KL-divergence of given vs known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">L1</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the mean absolute error between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">L2</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the L2 norm on the difference between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">L2Disparity</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
## <span class="component">MKLDivergence</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the mean KL-divergence of given vs known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">MSQ</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the mean absolute error between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">MSQRT</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the mean absolute error between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">Mabs</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the mean absolute error between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">MannWhitneyParity</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Performs a two-tailed Mann-Whitney U-test to check that the scores of sensitive-attributed nodes (ground truth) 
do not exhibit  higher or lower values compared to the rest. To do this, the test's U statistic is transformed so 
that value 1 indicates that the probability of sensitive-attributed nodes exhibiting higher values is the same as 
for lower values (50%). Value 0 indicates that either the probability of exhibiting only higher or only lower 
values is 100%. 
Known scores correspond to the binary sensitive attribute checking whether nodes are sensitive. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">MaxDifference</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the mean absolute error between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">Mistreatment</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes a disparate mistreatment assessment to test the fairness of given scores given 
that they are similarly evaluated by a measure of choice. The constructor args: 
 * *sensitive:* A binary graph signal that separates sensitive from non-sensitive nodes. 
 * *measure:* A supervised measure to compute disparate mistreament on. Default is AUC. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
known_score_signal, sensitive_signal = ... 
train, test = pg.split(known_scores, 0.8)  # 20% test set 
ranker = pg.LFPR() 
measure = pg.Mistreatment(known_scores, exclude=train, measure=pg.AUC) 
scores = ranker(train, sensitive=sensitive_signal) 
print(measure(scores)) 
```
## <span class="component">NDCG</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Provides evaluation of NDCG@k score between given and known scores. The constructor initializes the supervised measure with desired graph signal outcomes and the number of top scores. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
 * *k:* Optional. Calculates NDCG@k. If None (default), len(known_scores) is used.
## <span class="component">PPV</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the positive predictive value (precision). The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">PearsonCorrelation</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the Pearson correlation coefficient between given and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">RMabs</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the mean absolute error between scores and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">SpearmanCorrelation</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the Spearman correlation coefficient between given and known scores. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">TNR</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the false negative rate. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">TPR</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes the true positive rate (recall). The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">pRule</span>
<b class="parameters">Extends</b><br> *Supervised*<br><b class="parameters">About</b><br>
Computes an assessment of stochastic ranking fairness by obtaining the fractional comparison of average scores 
between sensitive-attributed nodes and the rest the rest. 
Values near 1 indicate full fairness (statistical parity), whereas lower values indicate disparate impact. 
Known scores correspond to the binary sensitive attribute checking whether nodes are sensitive. 
Usually, pRule > 80% is considered fair. The constructor initializes the supervised measure with desired graph signal outcomes. 
<br><b class="parameters">Parameters</b>

 * *known_scores:* The desired graph signal outcomes. 
 * *exclude:* Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed to determine which nodes to omit from the evaluation, for example because they were used for training. If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude` property at any time to alter this original value. Prefer using this behavior to avoid overfitting measure assessments.
## <span class="component">Conductance</span>
<b class="parameters">Extends</b><br> *Unsupervised*<br><b class="parameters">About</b><br>
Graph conductance (information flow) of scores. 
Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores, 
as per the formulation of [krasanakis2019linkauc] and calculates E[outgoing edges] / E[internal edges] of 
the fuzzy rank subgraph. To avoid potential optimization towards filling the whole graph, the measure is 
evaluated to infinity if either denominator *or* the nominator is zero (this means that whole connected components 
should not be extracted). 
If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Conductance. The constructor initializes the Conductance measure. 
<br><b class="parameters">Parameters</b>

 * *graph:* Optional. The graph on which to calculate the measure. If None (default) it is automatically extracted from graph signals passed for evaluation. 
 * *preprocessor:* Optional. Method to extract a scipy sparse matrix from a networkx graph. If None (default), pygrank.algorithms.utils.preprocessor is used with keyword arguments automatically extracted from the ones passed to this constructor, setting no normalization. 
 * *max_rank:* Optional. The maximum value scores can assume. To maintain a probabilistic formulation of conductance, this can be greater but not less than the maximum rank during evaluation. Default is 1. Pass algorithms through a normalization to ensure that this limit is not violated. 
 * *autofix:* Optional. If True, automatically normalizes scores by multiplying with max_rank / their maximum. If False (default) and the maximum score is greater than max_rank, an exception is thrown. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, seed_nodes, algorithm = ... 
algorithm = pg.Normalize(algorithm) 
scores = algorithm.rank(graph, seed_nodes) 
conductance = pg.Conductance().evaluate(scores) 
```
Example (same conductance):
```python 
conductance = pg.Conductance(autofix=True).evaluate(scores) 
```
## <span class="component">Density</span>
<b class="parameters">Extends</b><br> *Unsupervised*<br><b class="parameters">About</b><br>
Extension of graph density that accounts for node scores. 
Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores, 
as per the formulation of [krasanakis2019linkauc] and calculates E[internal edges] / E[possible edges] of 
the fuzzy rank subgraph. 
If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Density. The constructor initializes the Density measure. 
<br><b class="parameters">Parameters</b>

 * *graph:* Optional. The graph on which to calculate the measure. If None (default) it is automatically extracted from graph signals passed for evaluation. 
 * *preprocessor:* Optional. Method to extract a scipy sparse matrix from a networkx graph. If None (default), pygrank.algorithms.utils.preprocessor is used with keyword arguments automatically extracted from the ones passed to this constructor, setting no normalization. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, seed_nodes, algorithm = ... 
scores = algorithm.rank(graph, seed_nodes) 
density = pg.Density().evaluate(scores) 
```
## <span class="component">Modularity</span>
<b class="parameters">Extends</b><br> *Unsupervised*<br><b class="parameters">About</b><br>
Extension of modularity that accounts for node scores. The constructor initializes the Modularity measure with a sampling strategy that speeds up normal computations. 
<br><b class="parameters">Parameters</b>

 * *graph:* Optional. The graph on which to calculate the measure. If None (default) it is automatically extracted from graph signals passed for evaluation. 
 * *preprocessor:* Optional. Method to extract a scipy sparse matrix from a networkx graph. If None (default), pygrank.algorithms.utils.preprocessor is used with keyword arguments automatically extracted from the ones passed to this constructor, setting no normalization. 
 * *max_rank:* Optional. Default is 1. 
 * *max_positive_samples:* Optional. The number of nodes with which to compute modularity. These are sampled uniformly from all graph nodes. If this is greater than the number of graph nodes, all nodes are used and the measure is deterministic. However, calculation time is O(max_positive_samples<sup>2</sup>) and thus a trade-off needs to be determined of time vs approximation quality. Effectively, the value should be high enough for max_positive_samples<sup>2</sup> to be comparable to the number of graph edges. Default is 2000. 
 * *seed:* Optional. Makes the evaluation seeded, for example to use in tuning. Default is 0. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, seed_nodes, algorithm = ... 
scores = algorithm.rank(graph, seed_nodes) 
modularity = pg.Modularity(max_positive_samples=int(graph.number_of_edges()**0.5)).evaluate(scores) 
```
