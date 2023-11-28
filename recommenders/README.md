# Recommender Utilities

This package contains functions to simplify common tasks used when developing and evaluating recommender systems. A short description of the submodules is provided below.

# Contents

## [Datasets](datasets)

Datasets module includes helper functions for pulling different datasets and formatting them appropriately as well as utilities for splitting data for training / testing.

### Data Loading

There are dataloaders for several datasets. For example, the movielens module will allow you to load a dataframe in pandas or spark formats from the MovieLens dataset, with sizes of 100k, 1M, 10M, or 20M to test algorithms and evaluate performance benchmarks.

```python
df = movielens.load_pandas_df(size="100k")
```

### Splitting Techniques

Currently, three methods are available for splitting datasets. All of them support splitting by user or item and filtering out minimal samples (for instance users that have not rated enough items, or items that have not been rated by enough users).

- Random: this is the basic approach where entries are randomly assigned to each group based on the ratio desired
- Chronological: this uses provided timestamps to order the data and selects a cut-off time that will split the desired ratio of data to train before that time and test after that time
- Stratified: this is similar to random sampling, but the splits are stratified, for example if the datasets are split by user, the splitting approach will attempt to maintain the same ratio of items used in both training and test splits. The converse is true if splitting by item.

## [Evaluation](evaluation)

The evaluation submodule includes functionality for calculating common recommendation metrics directly in Python or in a Spark environment using PySpark.

Currently available metrics include:

- Root Mean Squared Error
- Mean Absolute Error
- R<sup>2</sup>
- Explained Variance
- Precision at K
- Recall at K
- Normalized Discounted Cumulative Gain at K
- Mean Average Precision at K
- Area Under Curve
- Logistic Loss

## [Models](models)

The models submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new recommender system approaches. A description of all the algorithms can be found on [this table](../README.md#algorithms). The following is a list of the algorithm utilities:

* Cornac
* DeepRec
  *  Convolutional Sequence Embedding Recommendation (CASER)
  *  Deep Knowledge-Aware Network (DKN)
  *  Extreme Deep Factorization Machine (xDeepFM)
  *  GRU
  *  LightGCN
  *  Next Item Recommendation (NextItNet)
  *  Short-term and Long-term Preference Integrated Recommender (SLi-Rec)
  *  Multi-Interest-Aware Sequential User Modeling (SUM)
* GeoIMC
* LightFM
* LightGBM
* NCF
* NewsRec
  * Neural Recommendation with Long- and Short-term User Representations (LSTUR)
  * Neural Recommendation with Attentive Multi-View Learning (NAML)
  * Neural Recommendation with Personalized Attention (NPA)
  * Neural Recommendation with Multi-Head Self-Attention (NRMS)
* Restricted Boltzmann Machines (RBM)
* Riemannian Low-rank Matrix Completion (RLRMC)
* Simple Algorithm for Recommendation (SAR)
* Self-Attentive Sequential Recommendation (SASRec)
* Sequential Recommendation Via Personalized Transformer (SSEPT)
* Surprise
* Term Frequency - Inverse Document Frequency (TF-IDF)
* Variational Autoencoders (VAE)
  * Multinomial
  * Standard
* Vowpal Wabbit (VW)
* Wide and Deep
* xLearn
  * Factorization Machine (FM)
  * Field-Aware FM (FFM)