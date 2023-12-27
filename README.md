# Evaluation Metrics for Recommendation Systems

[![DOI](https://zenodo.org/badge/692223902.svg)](https://zenodo.org/doi/10.5281/zenodo.10436717)

```
This repository contains the implementation of evaluation metrics for recommendation systems.
We have compared similarity, candidate generation, rating, ranking metrics performance on 5 different datasets - 
MovieLens 100k, MovieLens 1m, MovieLens 10m, Amazon Electronics Dataset and Amazon Movies and TV Dataset.
Summary of experiment with instructions on how to replicate this experiment can be find below.
```

### About Recommendations Models

Majority of this repository work is taken from - https://github.com/recommenders-team/recommenders

## Experiments Summary and Our Paper

### Cite Our Paper
```
@misc{jadon2023comprehensive,
      title={A Comprehensive Survey of Evaluation Techniques for Recommendation Systems}, 
      author={Aryan Jadon and Avinash Patil},
      year={2023},
      eprint={2312.16015},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

### Paper Link - https://arxiv.org/abs/2312.16015

## Summary of Experiments

###  Similarity Metrics
![similarity_metrics.png](docs%2Fsimilarity_metrics.png)

###  Candidate Generation Metrics
![candidate_generation_metrics.png](docs%2Fcandidate_generation_metrics.png)

### Rating Metrics
![rating_metrics.png](docs%2Frating_metrics.png)

### Ranking Metrics
![ranking_metrics.png](docs%2Franking_metrics.png)


## Replicating this Repository and Experiments

* **recommenders**: Folder containing the recommendations algorithms implementations.
* **similarity_metrics**: Folder containing scripts for running experiments of similarity metrics.
* **candidate_generation_metrics**: Folder containing scripts for running experiments of candidate generations metrics.
* **rating_metrics**: Folder containing scripts for running experiments of rating metrics.
* **ranking_metrics**: Folder containing scripts for running experiments of ranking metrics.


#### Creating Environment

Install the dependencies using requirements.txt

```bash
pip install -r requirements.txt
```
or 
```bash
conda env create -f environment.yml
```

#### Similarity Metrics Experiments

Run the Similarity Metrics Experiments using - 

```bash
chmod +x run_similarity_metrics_experiments.sh
./run_similarity_metrics_experiments.sh
```

#### Candidate Generation Metrics Experiments

Run the Candidate Generation Metrics Experiments using - 

```bash
chmod +x run_candidate_generation_metrics_experiments.sh
./run_candidate_generation_metrics_experiments.sh
```

#### Rating Metrics Experiments

Run the Rating Metrics Experiments using - 

```bash
chmod +x run_rating_metrics_experiments.sh
./run_rating_metrics_experiments.sh
```

#### Ranking Metrics Experiments

Run the Ranking Metrics Experiments using - 

```bash
chmod +x run_ranking_metrics_experiments.sh
./run_ranking_metrics_experiments.sh
```

