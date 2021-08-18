<center><h1>:notebook: Citations</h1></center>
Several research outcomes have been implemented and integrated in `pygrank`.
Please cite the respective publications in addition to this library.
<br>
<br>
For example, the seed oversampling method developed in *krasanakis2019boosted*
could be described in a LaTeX manuscript as 
*"we employ the seed oversampling (SeedO) post-processing \cite{krasanakis2019boosted}
implemented by the pygrank library \cite{pygrank}"*. Feel free to provide any other
description as long as they point to the library's paper and the corresponding publication(s).

## Method References

We hereby present a comprehensive match of
research outcomes to code concepts. For details on how to use respective
methods, please refer to the project's [documentation](documentation.md).
Reference names correspond to the list of publication bibtex entries
presented later on.

Instantiation or Usage | Method Name | Citation
--- | --- | --- 
`pygrank.algorithms.adhoc.AbsorbingWalks` | Absorbing Random Walk | wu2012learningadd
passing `lanczos=True` to graph filter constructors | Lanczos acceleration | susnjara2015accelerated
`pygrank.algorithms.oversampling.SeedOversampling(ranker)` | SeedO | krasanakis2019boosted
`pygrank.algorithms.oversampling.BoostedSeedOversampling(ranker)` | SeedBO | krasanakis2019boosted
`pygrank.algorithms.pagerank.PageRank(converge_to_eigenvectors=True)` | VenueRank | krasanakis2018venuerank
`pygrank.algorithms.postprocess.fairness.AdHocFairness(ranker,'fairwalk')` | FairWalk |rahman2019fairwalk
`pygrank.algorithms.postprocess.fairness.AdHocFairness(ranker,'O')` | LFPRO | tsioutsiouliklis2020fairness
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker, error_type="mabs", max_residual=0)` | FP | krasanakis2020fairconstr
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker, 0.8, 10, error_type="mabs", max_residual=0)` | CFP | krasanakis2020fairconstr
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker)` | FairEdit | krasanakis2020prioredit
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker, 0.8, 10)` | FairEditC | krasanakis2020prioredit
`pygrank.metrics.multigroup.LinkAUC(G, hops=1)` | LinkAUC | krasanakis2019linkauc
`pygrank.metrics.multigroup.LinkAUC(G, hops=2)` | HopAUC | krasanakis2020unsupervised
`pygrank.algorithms.utils.RankOrderConvergenceManager(alpha, confidence=0.99, criterion="fraction_of_walks")` | | krasanakis2020stopping
`pygrank.algorithms.utils.RankOrderConvergenceManager(alpha)` | | krasanakis2020stopping


## Publications
The publications that have led to the development of various aspects of
this library are presented in reverse chronological order.
```
@article{krasanakis2020unsupervised,
  title={Unsupervised evaluation of multiple node ranks by reconstructing local structures},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis},
  journal={Applied Network Science},
  volume={5},
  number={1},
  pages={1--32},
  year={2020},
  publisher={Springer}
}
```
```
@article{krasanakis2019boosted,
  title={Boosted seed oversampling for local community ranking},
  author={Krasanakis, Emmanouil and Schinas, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis and Symeonidis, Andreas},
  journal={Information Processing \& Management},
  pages={102053},
  year={2019},
  publisher={Elsevier}
}
```
```
@inproceedings{krasanakis2020stopping,
  title={Stopping Personalized PageRank without an Error Tolerance Parameter},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  year={2020},
  booktitle={ASONAM},
}
```
```
@inproceedings{krasanakis2020fairconstr,
  title={Applying Fairness Constraints on Graph Node Ranks under Personalization Bias},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  year={2020},
  booktitle={Complex Networks},
}
```
```
@inproceedings{krasanakis2019linkauc,
  title={LinkAUC: Unsupervised Evaluation of Multiple Network Node Ranks Using Link Prediction},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={3--14},
  year={2019},
  organization={Springer}
}
```
```
@inproceedings{krasanakis2018venuerank,
  title={VenueRank: Identifying Venues that Contribute to Artist Popularity},
  author={Krasanakis, Emmanouil and Schinas, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis and Mitkas, Pericles A},
  booktitle={ISMIR},
  pages={702--708},
  year={2018}
}
```

## Under Review
```
@inproceedings{krasanakis2020prioredit,
  title={Prior Signal Editing for Graph Filter Posterior Fairness Constraints},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  booktitle={preprint},
}
```

## Related
Additional publications whose methods are either fully 
or partially implemented in this library.
```
@article{tsioutsiouliklis2020fairness,
  title={Fairness-Aware Link Analysis},
  author={Tsioutsiouliklis, Sotiris and Pitoura, Evaggelia and Tsaparas, Panayiotis and Kleftakis, Ilias and Mamoulis, Nikos},
  journal={arXiv preprint arXiv:2005.14431},
  year={2020}
}
```
```
@inproceedings{rahman2019fairwalk,
  title={Fairwalk: Towards Fair Graph Embedding.},
  author={Rahman, Tahleen A and Surma, Bartlomiej and Backes, Michael and Zhang, Yang},
  booktitle={IJCAI},
  pages={3289--3295},
  year={2019}
}
```
```
@inproceedings{wu2012learning,
  title={Learning with Partially Absorbing Random Walks.},
  author={Wu, Xiao-Ming and Li, Zhenguo and So, Anthony Man-Cho and Wright, John and Chang, Shih-Fu},
  booktitle={NIPS},
  volume={25},
  pages={3077--3085},
  year={2012}
}
```
```
@article{susnjara2015accelerated,
  title={Accelerated filtering on graphs using lanczos method},
  author={Susnjara, Ana and Perraudin, Nathanael and Kressner, Daniel and Vandergheynst, Pierre},
  journal={arXiv preprint arXiv:1509.04537},
  year={2015}
}
```