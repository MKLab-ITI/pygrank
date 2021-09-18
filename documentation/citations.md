<center><h1>:notebook: Citations</h1></center>
Several research outcomes have been implemented and integrated in `pygrank`.
Please cite the respective publications in addition to this packge.

For example, the seed oversampling method developed in *krasanakis2019boosted*
could be described in a LaTeX manuscript as 
*"we employ the seed oversampling (SeedO) post-processing \cite{krasanakis2019boosted}
implemented by the pygrank package \cite{pygrank}"*. Feel free to provide any other
description as long as they point to the package's paper and the corresponding publication(s).

Do not forget to also cite dataset sources! Related instructions
can be accessed [here](datasets.md).

## Method References

We hereby present a comprehensive match of
research outcomes to code concepts. For details on how to use respective
methods, please refer to the project's [documentation](documentation.md).
Reference names correspond to the list of publication bibtex entries
presented later on.

Instantiation or Usage | Method Name | Citation
--- | --- | --- 
`PageRank` | Personalized PageRank | page1999pagerank
`HeatKernel` | Heat Kernel | chung2007heat
`AbsorbingWalks` | Absorbing Random Walk | wu2012learningadd
`GeneralizedGraphFilter` | Graph Filter | ortega2018graph
`GeneralizedGraphFilter(lanczos=dims)` (e.g. dims = 5) | Lanczos acceleration | susnjara2015accelerated
`SeedOversampling(ranker)` | SeedO | krasanakis2019boosted
`BoostedSeedOversampling(ranker)` | SeedBO | krasanakis2019boosted
`PageRank(converge_to_eigenvectors=True)` | VenueRank | krasanakis2018venuerank
`FairWalk(ranker)` | FairWalk |rahman2019fairwalk
`AdHocFairness(ranker,'O')` | LFPRO | tsioutsiouliklis2020fairness
`FairPersonalizer(ranker, error_type=Mabs, max_residual=0)` | FP | krasanakis2020fairconstr
`FairPersonalizer(ranker, 0.8, 10, error_type=Mabs, max_residual=0)` | CFP | krasanakis2020fairconstr
`FairPersonalizer(ranker, error_type=KLDivergence)` | FairEdit | krasanakis2020prioredit
`fairness.FairPersonalizer(ranker, error_type=KLDivergence, 0.8, 10)` | FairEditC | krasanakis2020prioredit
`LinkAssessment(graph, hops=1)` | LinkAUC | krasanakis2019linkauc
`LinkAssessment(graph, hops=2)` | HopAUC | krasanakis2020unsupervised
`ClusteringCoefficient(graph)` | LinkCC | krasanakis2020unsupervised
`PageRank(alpha, convergence=RankOrderConvergenceManager(alpha, confidence=0.99, criterion="fraction_of_walks"))` | | krasanakis2020stopping
`PageRank(alpha, RankOrderConvergenceManager(alpha))` | | krasanakis2020stopping


## Publications
Publications that have supported development of various aspects of
this library. These are presented in reverse chronological order.
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

## Preprints
```
@article{krasanakis2020prioredit,
  title={Prior Signal Editing for Graph Filter Posterior Fairness Constraints},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis and Symeonidis, Andreas},
  journal={arXiv:2108.12397},	
  year={2021}
}
```

## Related
Additional publications introducing methods implemented in this package.

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
```
@techreport{page1999pagerank,
  title={The PageRank citation ranking: Bringing order to the web.},
  author={Page, Lawrence and Brin, Sergey and Motwani, Rajeev and Winograd, Terry},
  year={1999},
  institution={Stanford InfoLab}
}
```
```
@article{chung2007heat,
  title={The heat kernel as the pagerank of a graph},
  author={Chung, Fan},
  journal={Proceedings of the National Academy of Sciences},
  volume={104},
  number={50},
  pages={19735--19740},
  year={2007},
  publisher={National Acad Sciences}
}
```
```
@article{ortega2018graph,
  title={Graph signal processing: Overview, challenges, and applications},
  author={Ortega, Antonio and Frossard, Pascal and Kova{\v{c}}evi{\'c}, Jelena and Moura, Jos{\'e} MF and Vandergheynst, Pierre},
  journal={Proceedings of the IEEE},
  volume={106},
  number={5},
  pages={808--828},
  year={2018},
  publisher={IEEE}
}
```
