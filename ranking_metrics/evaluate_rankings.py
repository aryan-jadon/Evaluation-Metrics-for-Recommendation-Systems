import gc
import warnings
import logging
import sys
import surprise
import cornac
import pyspark
import tensorflow as tf
import torch
from recommenders.datasets import movielens
from recommenders.utils.general_utils import get_number_processors
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.utils.gpu_utils import get_cuda_version, get_cudnn_version
from ranking_metrics.benchmark_utils import *

# Set log levels to prevent unnecessary logging
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

print(f"System version: {sys.version}")
print(f"Number of cores: {get_number_processors()}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Surprise version: {surprise.__version__}")
print(f"Cornac version: {cornac.__version__}")
print(f"PySpark version: {pyspark.__version__}")
print(f"CUDA version: {get_cuda_version()}")
print(f"CuDNN version: {get_cudnn_version()}")
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Initialize Spark with limited memory usage
spark = pyspark.sql.SparkSession.builder \
    .appName("PySpark-Ranking-Experiment") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Movielens data size: 100k, 1m, 10m, or 20m
data_sizes = ["100k", "1m", "10m", "20m"]
algorithms = [
    "als",
    "sar",
    "svd",
    "ncf",
    "bpr",
    "bivae",
    "lightgcn"
]

environments = {
    "als": "pyspark",
    "sar": "python_cpu",
    "svd": "python_cpu",
    "bpr": "python_cpu",
    "ncf": "python_gpu",
    "bivae": "python_gpu",
    "lightgcn": "python_gpu",
}

metrics = {
    "als": ["ranking"],
    "sar": ["ranking"],
    "svd": ["ranking"],
    "ncf": ["ranking"],
    "bpr": ["ranking"],
    "bivae": ["ranking"],
    "lightgcn": ["ranking"]
}

als_params = {
    "rank": 10,
    "maxIter": 20,
    "implicitPrefs": False,
    "alpha": 0.1,
    "regParam": 0.05,
    "coldStartStrategy": "drop",
    "nonnegative": False,
    "userCol": DEFAULT_USER_COL,
    "itemCol": DEFAULT_ITEM_COL,
    "ratingCol": DEFAULT_RATING_COL,
}

sar_params = {
    "similarity_type": "jaccard",
    "time_decay_coefficient": 30,
    "time_now": None,
    "timedecay_formula": True,
    "col_user": DEFAULT_USER_COL,
    "col_item": DEFAULT_ITEM_COL,
    "col_rating": DEFAULT_RATING_COL,
    "col_timestamp": DEFAULT_TIMESTAMP_COL,
}

svd_params = {
    "n_factors": 150,
    "n_epochs": 15,
    "lr_all": 0.005,
    "reg_all": 0.02,
    "random_state": SEED,
    "verbose": False
}

ncf_params = {
    "model_type": "NeuMF",
    "n_factors": 4,
    "layer_sizes": [16, 8, 4],
    "n_epochs": 15,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "verbose": 10
}

bpr_params = {
    "k": 200,
    "max_iter": 200,
    "learning_rate": 0.01,
    "lambda_reg": 1e-3,
    "seed": SEED,
    "verbose": False
}

bivae_params = {
    "k": 100,
    "encoder_structure": [200],
    "act_fn": "tanh",
    "likelihood": "pois",
    "n_epochs": 500,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "seed": SEED,
    "use_gpu": True,
    "verbose": False
}

lightgcn_param = {
    "model_type": "lightgcn",
    "n_layers": 3,
    "batch_size": 1024,
    "embed_size": 64,
    "decay": 0.0001,
    "epochs": 20,
    "learning_rate": 0.005,
    "eval_epoch": 5,
    "top_k": DEFAULT_K,
    "metrics": ["recall", "ndcg", "precision", "map"],
    "save_model": False,
    "MODEL_DIR": ".",
}

params = {
    "als": als_params,
    "sar": sar_params,
    "svd": svd_params,
    "ncf": ncf_params,
    "bpr": bpr_params,
    "bivae": bivae_params,
    "lightgcn": lightgcn_param,
}

prepare_training_data = {
    "als": prepare_training_als,
    "sar": prepare_training_sar,
    "svd": prepare_training_svd,
    "ncf": prepare_training_ncf,
    "bpr": prepare_training_cornac,
    "bivae": prepare_training_cornac,
    "lightgcn": prepare_training_lightgcn,
}

prepare_metrics_data = {
    "als": lambda train, test: prepare_metrics_als(train, test)
}

trainer = {
    "als": lambda params, data: train_als(params, data),
    "sar": lambda params, data: train_sar(params, data),
    "svd": lambda params, data: train_svd(params, data),
    "ncf": lambda params, data: train_ncf(params, data),
    "bpr": lambda params, data: train_bpr(params, data),
    "bivae": lambda params, data: train_bivae(params, data),
    "lightgcn": lambda params, data: train_lightgcn(params, data),
}

ranking_predictor = {
    "als": lambda model, test, train: recommend_k_als(model, test, train),
    "sar": lambda model, test, train: recommend_k_sar(model, test, train),
    "svd": lambda model, test, train: recommend_k_svd(model, test, train),
    "ncf": lambda model, test, train: recommend_k_ncf(model, test, train),
    "bpr": lambda model, test, train: recommend_k_cornac(model, test, train),
    "bivae": lambda model, test, train: recommend_k_cornac(model, test, train),
    "lightgcn": lambda model, test, train: recommend_k_lightgcn(model, test, train),
}

ranking_evaluator = {
    "als": lambda test, predictions, k: ranking_metrics_pyspark(test, predictions, k),
    "sar": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "svd": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "ncf": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bpr": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "bivae": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
    "lightgcn": lambda test, predictions, k: ranking_metrics_python(test, predictions, k),
}


def generate_summary(data, algo, k, train_time, time_ranking, ranking_metrics):
    summary = {"Data": data,
               "Algo": algo,
               "K": k,
               "Train time (s)": train_time,
               "Recommending time (s)": time_ranking}
    print(ranking_metrics)
    summary.update(ranking_metrics)
    return summary


algosummary = {}

for data_size in data_sizes:
    try:
        print("Working on Data Size" + str(data_size))
        # Load the dataset
        df = movielens.load_pandas_df(
            size=data_size,
            header=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]
        )
        print("Size of Movielens {}: {}".format(data_size, df.shape))

        # Split the dataset
        df_train, df_test = python_stratified_split(df,
                                                    ratio=0.75,
                                                    min_rating=1,
                                                    filter_by="item",
                                                    col_user=DEFAULT_USER_COL,
                                                    col_item=DEFAULT_ITEM_COL
                                                    )

        # Loop through the algos
        for algo in algorithms:
            try:
                print(f"\nComputing {algo} algorithm on Movielens {data_size}")

                # Data prep for training set
                train = prepare_training_data.get(algo, lambda x, y: (x, y))(df_train, df_test)

                # Get model parameters
                model_params = params[algo]

                # Train the model
                model, time_train = trainer[algo](model_params, train)
                print(f"Training time: {time_train}s")

                # Predict and evaluate
                train, test = prepare_metrics_data.get(algo, lambda x, y: (x, y))(df_train, df_test)

                if "ranking" in metrics[algo]:
                    # Predict for ranking
                    top_k_scores, time_ranking = ranking_predictor[algo](model, test, train)
                    print(f"Ranking prediction time: {time_ranking}s")

                    # Evaluate for ranking
                    rankings = ranking_evaluator[algo](test, top_k_scores, DEFAULT_K)
                else:
                    rankings = None
                    time_ranking = np.nan

                # Record results
                algosummary[algo] = generate_summary(data_size,
                                                     algo,
                                                     DEFAULT_K,
                                                     time_train,
                                                     time_ranking,
                                                     rankings)

                algosummary[algo]["F1@K"] = 2 * (algosummary[algo]["Precision@k"] * algosummary[algo]["Recall@k"]) / (
                        algosummary[algo]["Precision@k"] + algosummary[algo]["Recall@k"])

                print("#" * 100)
                print("Complete Summary on DataSet")
                print(algosummary)
                print("#" * 100)

                df = pd.DataFrame(algosummary)
                file_name = "ranking_results_{algo}_{size}.xlsx".format(algo=algo, size=data_size)
                df.to_excel(file_name)

                del train
                del test
                del model_params
                del model
                del time_train
                # Garbage collection
                gc.collect()
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)

print("\nComputation finished")
