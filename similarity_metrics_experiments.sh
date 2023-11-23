# ==========================================================================================================================================
### Workflow for amazon dataset electronics experiments

python -m similarity_metrics.parse.parse_json data/meta_Electronics.json.gz data/electronics.csv
python -m similarity_metrics.prep.prep_node_relationship data/electronics.csv data/electronics_relationships.csv
python -m similarity_metrics.prep.prep_edges data/electronics_relationships.csv data/electronics_edges.csv
python -m similarity_metrics.prep.train_val_split data/electronics_edges.csv 0.33
python -m similarity_metrics.prep.prep_graph_samples data/electronics_edges_train.edgelist data/electronics_sequences.npy electronics


python -m similarity_metrics.ml.train_torch_embedding_cosine_similarity data/electronics_sequences.npy data/electronics_edges_val.csv  data/electronics_edges_val.csv 128 4
python -m similarity_metrics.ml.train_torch_embedding_cosine_similarity data/electronics_sequences.npy data/electronics_edges_val.csv  data/electronics_edges_val.csv 128 4