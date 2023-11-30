# ==========================================================================================================================================
### Workflow for amazon dataset electronics experiments

## Data Preparation for Benchmarking Experiment
python -m similarity_metrics.parse.parse_json similarity_metrics/data/meta_Electronics.json.gz similarity_metrics/data/electronics.csv
python -m similarity_metrics.prep.prep_node_relationship similarity_metrics/data/electronics.csv similarity_metrics/data/electronics_relationships.csv
python -m similarity_metrics.prep.prep_edges similarity_metrics/data/electronics_relationships.csv similarity_metrics/data/electronics_edges.csv
python -m similarity_metrics.prep.train_val_split similarity_metrics/data/electronics_edges.csv 0.33
python -m similarity_metrics.prep.prep_graph_samples similarity_metrics/data/electronics_edges_train.edgelist similarity_metrics/data/electronics_sequences.npy electronics

## Benchmarking Similarity Metrics Experiments
python -m similarity_metrics.torch_embedding_cosine_similarity similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_adjusted_cosine_similarity similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_chebyshev_distance similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_euclidean_distance similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_hamming_distance similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_jaccard_index similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_manhattan_distance similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_pearson_correlation_coefficient similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
python -m similarity_metrics.torch_embedding_spearman_rank_order_correlation_coefficient similarity_metrics/data/electronics_sequences.npy similarity_metrics/data/electronics_edges_val.csv  similarity_metrics/data/electronics_edges_val.csv 128 4
# ==========================================================================================================================================