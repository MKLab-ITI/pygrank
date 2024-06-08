# Postprocessors
The following postprocessors can be imported from the package `pygrank.algorithms.postprocess`.
Constructor details are provided, including arguments inherited from and passed to parent classes.
 
## <span class="component">AdHocFairness</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Adjusts node scores so that the sum of sensitive nodes is moved closer to the sum of non-sensitive ones based on 
ad hoc literature assumptions about how unfairness is propagated in graphs. The constructor initializes the fairness-aware postprocessor. 
<br><b class="parameters">Parameters</b>

 * *ranker:* The base ranking algorithm. 
 * *method:* The method with which to adjust weights. If "O" (default) an optimal gradual adjustment is performed [tsioutsiouliklis2020fairness]. If "B" node scores are weighted according to whether the nodes are sensitive, so that the sum of sensitive node scores becomes equal to the sum of non-sensitive node scores [tsioutsiouliklis2020fairness]. 
 * *eps:* A small value to consider rank redistribution to have converged. Default is 1.E-12.
## <span class="component">BoostedSeedOversampling</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Iteratively performs seed oversampling and combines found ranks by weighting them with a Boosting scheme. The constructor initializes the class with a base ranker and the boosting scheme'personalization parameters. 
<br><b class="parameters">Parameters</b>

 * *ranker:* The base ranker instance. 
 * *objective:* Optional. Can be either "partial" (default) or "naive". 
 * *oversample_from_iteration:* Optional. Can be either "previous" (default) to oversample the ranks of the previous iteration or "original" to always ovesample the given personalization. 
 * *weight_convergence:* Optional.  A ConvergenceManager that helps determine whether the weights placed on boosting iterations have converged. If None (default), initialized with ConvergenceManager(error_type=pyrgank.MaxDifference, tol=0.001, max_iters=100) 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, seed_nodes = ... 
algorithm = pg.BoostedSeedOversampling(pg.PageRank(alpha=0.99)) 
ranks = algorithm.rank(graph, personalization={1 for v in seed_nodes}) 
```
## <span class="component">FairPersonalizer</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
A personalization editing scheme that aims to edit graph signal priors (i.e. personalization) to produce 
fairness-aware posteriors satisfying disparate impact constraints in terms of pRule. The constructor instantiates a personalization editing scheme that trains towards optimizing 
retain_rank_weight*error_type(original scores, editing-induced scores) 
+ pRule_weight*min(induced score pRule, target_pRule) 
<br><b class="parameters">Parameters</b>

 * *ranker:* The base ranking algorithm. 
 * *target_pRule:* Up to which value should pRule be improved. pRule values greater than this are not penalized further. 
 * *retain_rank_weight:* Can be used to penalize deviations from original posteriors due to editing. Use the default value 1 unless there is a specific reason to scale the error. Higher values correspond to tighter maintenance of original posteriors, but may not improve fairness as much. 
 * *pRule_weight:* Can be used to penalize low pRule values. Either use the default value 1 or, if you want to place most emphasis on pRule maximization (instead of trading-off between fairness and posterior preservation) 10 is a good empirical starting point. 
 * *error_type:* The supervised measure used to penalize deviations from original posterior scores. pygrank.KLDivergence (default) uses is used in [krasanakis2020prioredit]. pygrank.Error is used by the earlier [krasanakis2020fairconstr]. The latter does not induce fairness as well on average, but is sometimes better for specific graphs. 
 * *parameter_buckets:* How many sets of parameters to be used to . Default is 1. More parameters could be needed to to track, but running time scales **exponentially** to these (with base 4). 
 * *max_residual:* An upper limit on how much the original personalization is preserved, i.e. a fraction of it in the range [0, max_residual] is preserved. Default is 1 and is introduced by [krasanakis2020prioredit], but 0 can be used for exact replication of [krasanakis2020fairconstr]. 
 * *parity_type:* The type of fairness measure to be optimized. If "impact" (default) the pRule is optimized, if "TPR" or "TNR" the TPR and TNR parity between sensitive and non-sensitive nodes is optimized respectively, if "mistreatment" the AM of TPR and TNR parity is optimized. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, sensitive, algorithm = ... # sensitive is a second graph signal 
algorithm = pg.FairPersonalizer(algorithm, .8, pRule_weight=10) # tries to force (weight 10) pRule to be at least 80% 
ranks = algorithm.rank(graph, personalization, sensitive=sensitive) 
```
Example (treats class imbalanace):
```python 
graph, personalization, algorithm = ... 
ranks = algorithm.rank(graph, personalization, sensitive=personalization) 
```
## <span class="component">FairWalk</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Adjusting graph convolutions to perform fair random walking [rahman2019fairwalk]. The constructor initializes Fairwalk given a base ranker. **This explicitly assumes immutability** of graphs. If you edit 
graphs also clear the dictionary where preprocessed graphs are inputted by calling `pygrank.fairwalk.reweights.clear().` 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. If None (default), a Tautology() ranker is created.
## <span class="component">LinearSweep</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Applies a sweep procedure that subtracts non-personalized ranks from personalized ones. The constructor initializes the sweep procedure. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. 
 * *uniform_ranker:* Optional. The ranker instance used to perform non-personalized ranking. If None (default) the base ranker is used. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm = ... 
algorithm = pg.LinearSweep(algorithm) # subtracts from node scores a uniform ranker's non-personalized outcome 
ranks = algorithm.rank(graph, personalization 
```
Example with different rankers:
```python 
graph, personalization, algorithm, uniform_ranker = ... 
algorithm = pg.LinearSweep(algorithm, uniform_ranker=uniform_ranker) 
ranks = algorithm.rank(graph, personalization) 
```
Example (same outcome):
```python 
graph, personalization, uniform_ranker, algorithm = ... 
ranks = pg.Threshold(uniform_ranker).transform(algorithm.rank(graph, personalization)) 
```
## <span class="component">MabsMaintain</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Forces node ranking posteriors to have the same mean absolute value as prior inputs. The constructor initializes the postprocessor with a base ranker instance. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. If None (default), a Tautology() ranker is created.
## <span class="component">Normalize</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Normalizes ranks by dividing with their maximal value. The constructor initializes the class with a base ranker instance. Args are automatically filled in and 
re-ordered if at least one is provided. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 
 * *method:* Optional. Divide ranks either by their "max" (default) or by their "sum" or make the lie in the "range" [0,1] by subtracting their mean before diving by their max. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm = ... 
algorithm = pg.Normalize(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0 
ranks = algorithm.rank(graph, personalization) 
```
Example (same outcome, simpler one-liner):
```python 
ranks = pg.Normalize(0.5).transform(algorithm.rank(graph, personalization)) 
```
## <span class="component">Ordinals</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Converts ranking outcome to ordinal numbers. 
The highest rank is set to 1, the second highest to 2, etc. The constructor initializes the class with a base ranker instance. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm = ... 
algorithm = pg.Ordinals(algorithm) 
ranks = algorithm.rank(graph, personalization) 
```
Example (same outcome, simpler one-liner):
```python 
ranks = pg.Ordinals().transform(algorithm.rank(graph, personalization)) 
```
## <span class="component">SeedOversampling</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Performs seed oversampling on a base ranker to improve the quality of predicted seeds. The constructor initializes the class with a base ranker. 
<br><b class="parameters">Parameters</b>

 * *ranker:* The base ranker instance. 
 * *method:* Optional. Can be "safe" (default) to oversample based on the ranking scores of a preliminary base ranker run or "neighbors" to oversample the neighbors of personalization nodes. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, seed_nodes = ... 
algorithm = pg.SeedOversampling(pg.PageRank(alpha=0.99)) 
ranks = algorithm.rank(graph, personalization={1 for v in seed_nodes}) 
```
## <span class="component">SeparateNormalization</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Performs different normalizations between two different groups of nodes. 
Intended use is in implementing algorithms like HITS. The constructor initializes the postprocessor. 
<br><b class="parameters">Parameters</b>

 * *separator:* A graph signal (preferred) or data structure convertible to one. Is meant to hold binary node scores, but other values in the range [0,1] are allowed and interpolated. 
 * *ranker:* Optional. The base ranker instance. If None (default) a Tautology ranker is used.
## <span class="component">Sequential</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
## <span class="component">Subgraph</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Extracts induced subgraphs for non-zero node scores and places those scores in new signals on it. The constructor initializes the postprocessor with a base ranker. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm = ... 
algorithm = pg.Subgraph(pg.Top(algorithm, 10)) 
top_10_subgraph = algorithm(graph, personalization).graph 
```
Example (same result):
```python 
algorithm = algorithm >> pg.Top(10) >> pg.Subgraph() 
```
## <span class="component">Supergraph</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Reverts to full graphs from which `Subgraph` departed. The constructor initializes the postprocessor with a base ranker. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm, test = ... 
algorithm = algorithm >> pg.Top(10) >> pg.Threshold() >> pg.Subgraph() >> pg.PageRank() >> pg.Supergraph() 
top_10_reranked = algorithm(graph, personalization)  # top 10 non-zeroes ranked in their induced subgraph 
print(pg.AUC(pg.to_signal(graph, test))(top_10_reranked))  # supergraph has returned to the original graph 
```
## <span class="component">Sweep</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Applies a sweep procedure that divides personalized node ranks by corresponding non-personalized ones. The constructor initializes the sweep procedure. 
<br><b class="parameters">Parameters</b>

 * *ranker:* The base ranker instance. 
 * *uniform_ranker:* Optional. The ranker instance used to perform non-personalized ranking. If None (default) the base ranker is used. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm = ... 
algorithm = pg.Sweep(algorithm) # divides node scores by a uniform ranker's non-personalized outcome 
ranks = algorithm.rank(graph, personalization 
```
Example with different rankers:
```python 
graph, personalization, algorithm, uniform_ranker = ... 
algorithm = pg.Sweep(algorithm, uniform_ranker=uniform_ranker) 
ranks = algorithm.rank(graph, personalization) 
```
Example (same outcome):
```python 
graph, personalization, uniform_ranker, algorithm = ... 
ranks = pg.Threshold(uniform_ranker).transform(algorithm.rank(graph, personalization)) 
```
## <span class="component">Tautology</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Returns ranks as-are. 
Can be used as a baseline against which to compare other postprocessors or graph filters. The constructor initializes the Tautology postprocessor with a base ranker. 
<br><b class="parameters">Parameters</b>

 * *ranker:* The base ranker instance. If None (default), this works as a base ranker that returns a copy of personalization signals as-are or a conversion of backend primitives into signals.
## <span class="component">Threshold</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Converts ranking outcome to binary values based on a threshold value. The constructor initializes the Threshold postprocessing scheme. Args are automatically filled in and 
re-ordered if at least one is provided. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 
 * *threshold:* Optional. The maximum numeric value required to output rank 0 instead of 1. If "gap" then its value is automatically determined based on the maximal percentage increase between consecutive ranks. Default is 0. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm = ... 
algorithm = pg.Threshold(algorithm, 0.5) # sets ranks >= 0.5 to 1 and lower ones to 0 
ranks = algorithm.rank(graph, personalization) 
```
Example (same outcome):
```python 
ranks = pg.Threshold(0.5).transform(algorithm.rank(graph, personalization)) 
```
Example (binary conversion):
```python 
graph = ... 
binary = pg.Threshold(0).transform(pg.to_signal(graph, [0, 0.1, 0, 1]))  # creates [0, 1, 0, 1] ranks 
```
## <span class="component">Top</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Keeps the top ranks as are and converts other ranks to zero. The constructor initializes the class with a  base ranker instance and number of top examples. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 
 * *fraction_of_training:* Optional. If 1 (default) or greater, keep that many top-scored nodes. If less than 1, it finds a corresponding fraction of the the graph signal to zero (e.g. for 0.5 set the lower half node scores to zero). 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, group, algorithm = ... 
training, test = pg.split(pg.to_signal(graph, group)) 
ranks = pg.Normalize(algorithm, "sum").rank(training) 
ranks = ranks*(1-training) 
top5 = pg.Threshold(pg.Top(5))(ranks)  # top5 ranks converted to 1, others to 0 
print(pg.TPR(test, exclude=training)(top5)) 
```
## <span class="component">Transformer</span>
<b class="parameters">Extends</b><br> *Postprocessor*<br><b class="parameters">About</b><br>
Applies an element-by-element transformation on a graph signal based on a given expression. The constructor initializes the class with a base ranker instance. Args are automatically filled in and 
re-ordered if at least one is provided. 
<br><b class="parameters">Parameters</b>

 * *ranker:* Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified. 
 * *expr:* Optional. A lambda expression to apply on each element. The transformer will automatically try to apply it on the backend array representation of the graph signal first, so prefer pygrank's backend functions for faster computations. For example, backend.exp (default) should be preferred instead of math.exp, because the former can directly parse numpy arrays, tensors, etc. 

<b class="parameters">Example</b>
```python 
import pygrank as pg 
graph, personalization, algorithm = ... 
r1 = pg.Normalize(algorithm, "sum").rank(graph, personalization) 
r2 = pg.Transformer(algorithm, lambda x: x/pg.sum(x)).rank(graph, personalization) 
print(pg.Mabs(r1)(r2)) 
```
