# Parametric and structure-aware information theory

This repository accompanies the article *Parametric and structure-aware information theory for multi-scale analysis of composition*.

## File structure
+ `data`: Data files for the case studies.
+ `tools`
    + `core.py`: Calculate parametric and structure-aware entropy, divergence, and Bregman information.
    + `clustering.py`: *k*-means-style clustering to maximise Bregman information explained by the partition.
+ `ouptuts`: Data files of results.
+ `Fig1.ipynb`: Replicate Fig. 1 that demonstrates how the divergence is sensitive to $\alpha$ and **Z**.
+ `counterexample_ricotta_szeidl_2006.ipynb`: Counterexample to [Ricotta and Szeidl (2006)](https://www.sciencedirect.com/science/article/abs/pii/S004058090600075X), Theorem 2.
+ `region_of_concavity.ipynb`: Map out the region of strict concavity for $\alpha=3$.
+ `nonconvex_region_of_concavity.ipynb`: Example to show that the region of strict concavity is not necessarily convex for $\alpha=4$.
+ `OT_comparison.ipynb`: Experiments comparing runtime against optimal transport to replicate Fig. 2.
+ `occupations_preprocessing.ipynb`: Pre-processing England and Wales occupation data to select tradable occupations.
+ `occupations_analysis.ipynb`: Analysis of the geography of occupation composition in England and Wales to replicate Fig. 3 and Table 1.
+ `rutor_glacier_analysis.ipynb`: Analysis of functional and taxonomic $\beta$-diversity in the Rutor glacier to replicat Fig. 4.

## Acknowledgements
Thank you Renaud Lambiotte and Karel Devriendt.