<center><h1>:notebook: Citations</h1></center>
Several research outcomes have been implemented and integrated in `pygrank`.
In addition to this package, please cite related publications, for example
by modifying automatically generated descriptions (see below). 

Do not forget to also cite dataset sources! Related instructions
can be accessed [here](datasets.md).

## Automated Algorithm Citations

The `NodeRanking.cite()` method can be used obtain 
automatically-generated descriptions of algorithms, including
publication citations. Reference names correspond to the list of
publication bibtex entries presented in the rest of the document.

For example, the following snippet defines a node ranking algorithm
and retrieves a textual description to be pasted in LaTeX documents
after adding respective publications to the bibliography.

```python
>>> import pygrank as pg
>>> algorithm = pg.BoostedSeedOversampling(pg.PageRank())
>>> print(algorithm.cite())
personalized PageRank \cite{page1999pagerank} with restart probability 0.15 and iterative partial boosted seed oversampling of previous node scores \cite{krasanakis2019boosted} postprocessor
```

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

```
@misc{krasanakis2021pygrank,
      title={pygrank: A Python Package for Graph Node Ranking}, 
      author={Emmanouil Krasanakis, Symeon Papadopoulos, Ioannis Kompatsiaris, Andreas Symeonidis},
      year={2021},
      eprint={2110.09274},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
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

```
@inproceedings{yu2021chebyshev,
  title={Chebyshev Accelerated Spectral Clustering},
  author={Yu, Tianyu and Zhao, Yonghua and Huang, Rongfeng and Liu, Shifang and Zhang, Xinyin},
  booktitle={Proceedings of the 14th ACM International Conference on Web Search and Data Mining},
  pages={247--255},
  year={2021}
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
@inproceedings{andersen2007local,
  title={Local partitioning for directed graphs using PageRank},
  author={Andersen, Reid and Chung, Fan and Lang, Kevin},
  booktitle={International Workshop on Algorithms and Models for the Web-Graph},
  pages={166--178},
  year={2007},
  organization={Springer}
}
```