import os
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory
import surprise
import cornac
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import FloatType, IntegerType, LongType

from recommenders.utils.timer import Timer
from recommenders.utils.constants import (
    COL_DICT,
    DEFAULT_K,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_TIMESTAMP_COL,
    SEED,
)
from recommenders.models.sar import SAR
from recommenders.models.surprise.surprise_utils import (
    predict,
    compute_ranking_predictions,
)
from recommenders.models.cornac.cornac_utils import predict_ranking

# evaluation metrics
from recommenders.evaluation.python_evaluation import (
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    mrr_at_k,
    arhr_at_k,
    average_recall_at_k,
    average_precision_at_k
)
from recommenders.evaluation.python_evaluation import rmse, mae, rsquared, exp_var, mse, mape
from recommenders.utils.spark_utils import start_or_get_spark
from recommenders.evaluation.spark_evaluation import (
    SparkRatingEvaluation,
    SparkRankingEvaluation,
)

# skip this import if we are not in a GPU environment
try:
    from fastai.collab import collab_learner, CollabDataBunch
    from recommenders.models.fastai.fastai_utils import (
        cartesian_product,
        score,
    )
except ImportError:
    pass

# skip this import if we are not in a GPU environment

from recommenders.models.deeprec.deeprec_utils import prepare_hparams
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF

# skip this import if we are not in a GPU environment
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as ExperimentNCFDataset

# Helpers
tmp_dir = TemporaryDirectory()
TRAIN_FILE = os.path.join(tmp_dir.name, "df_train.csv")
TEST_FILE = os.path.join(tmp_dir.name, "df_test.csv")


def prepare_training_als(train, test):
    schema = StructType(
        (
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, LongType()),
        )
    )
    spark = start_or_get_spark()
    return spark.createDataFrame(train, schema).cache()


def train_als(params, data):
    symbol = ALS(**params)
    with Timer() as t:
        model = symbol.fit(data)
    return model, t


def prepare_metrics_als(train, test):
    schema = StructType(
        (
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, LongType()),
        )
    )
    spark = start_or_get_spark()
    return (
        spark.createDataFrame(train, schema).cache(),
        spark.createDataFrame(test, schema).cache(),
    )


def predict_als(model, test):
    with Timer() as t:
        preds = model.transform(test)
    return preds, t


def recommend_k_als(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        # Get the cross join of all user-item pairs and score them.
        users = train.select(DEFAULT_USER_COL).distinct()
        items = train.select(DEFAULT_ITEM_COL).distinct()
        user_item = users.crossJoin(items)
        dfs_pred = model.transform(user_item)

        # Remove seen items
        dfs_pred_exclude_train = dfs_pred.alias("pred").join(
            train.alias("train"),
            (dfs_pred[DEFAULT_USER_COL] == train[DEFAULT_USER_COL])
            & (dfs_pred[DEFAULT_ITEM_COL] == train[DEFAULT_ITEM_COL]),
            how="outer",
        )
        topk_scores = dfs_pred_exclude_train.filter(
            dfs_pred_exclude_train["train." + DEFAULT_RATING_COL].isNull()
        ).select(
            "pred." + DEFAULT_USER_COL,
            "pred." + DEFAULT_ITEM_COL,
            "pred." + DEFAULT_PREDICTION_COL,
        )
    return topk_scores, t


def prepare_training_svd(train, test):
    reader = surprise.Reader("ml-100k", rating_scale=(1, 5))
    return surprise.Dataset.load_from_df(
        train.drop(DEFAULT_TIMESTAMP_COL, axis=1), reader=reader
    ).build_full_trainset()


def train_svd(params, data):
    model = surprise.SVD(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def predict_svd(model, test):
    with Timer() as t:
        preds = predict(
            model,
            test,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
        )
    return preds, t


def recommend_k_svd(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = compute_ranking_predictions(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=remove_seen,
        )
    return topk_scores, t


def prepare_training_fastai(train, test):
    data = train.copy()
    data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].astype("str")
    data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].astype("str")
    data = CollabDataBunch.from_df(
        data,
        user_name=DEFAULT_USER_COL,
        item_name=DEFAULT_ITEM_COL,
        rating_name=DEFAULT_RATING_COL,
        valid_pct=0,
    )
    return data


def train_fastai(params, data):
    model = collab_learner(
        data, n_factors=params["n_factors"], y_range=params["y_range"], wd=params["wd"]
    )
    with Timer() as t:
        model.fit_one_cycle(cyc_len=params["epochs"], max_lr=params["max_lr"])
    return model, t


def prepare_metrics_fastai(train, test):
    data = test.copy()
    data[DEFAULT_USER_COL] = data[DEFAULT_USER_COL].astype("str")
    data[DEFAULT_ITEM_COL] = data[DEFAULT_ITEM_COL].astype("str")
    return train, data


def predict_fastai(model, test):
    with Timer() as t:
        preds = score(
            model,
            test_df=test,
            user_col=DEFAULT_USER_COL,
            item_col=DEFAULT_ITEM_COL,
            prediction_col=DEFAULT_PREDICTION_COL,
        )
    return preds, t


def recommend_k_fastai(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        total_users, total_items = model.data.train_ds.x.classes.values()
        total_items = total_items[1:]
        total_users = total_users[1:]
        test_users = test[DEFAULT_USER_COL].unique()
        test_users = np.intersect1d(test_users, total_users)
        users_items = cartesian_product(test_users, total_items)
        users_items = pd.DataFrame(
            users_items, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL]
        )
        training_removed = pd.merge(
            users_items,
            train.astype(str),
            on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL],
            how="left",
        )
        training_removed = training_removed[
            training_removed[DEFAULT_RATING_COL].isna()
        ][[DEFAULT_USER_COL, DEFAULT_ITEM_COL]]
        topk_scores = score(
            model,
            test_df=training_removed,
            user_col=DEFAULT_USER_COL,
            item_col=DEFAULT_ITEM_COL,
            prediction_col=DEFAULT_PREDICTION_COL,
            top_k=top_k,
        )
    return topk_scores, t


def prepare_training_ncf(df_train, df_test):
    train = df_train.sort_values(["userID"], axis=0, ascending=[True])
    test = df_test.sort_values(["userID"], axis=0, ascending=[True])
    test = test[df_test["userID"].isin(train["userID"].unique())]
    test = test[test["itemID"].isin(train["itemID"].unique())]
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)
    return ExperimentNCFDataset(
        train_file=TRAIN_FILE,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        seed=SEED,
    )


def train_ncf(params, data):
    model = NCF(n_users=data.n_users, n_items=data.n_items, **params)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_ncf(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        users, items, preds = [], [], []
        item = list(train[DEFAULT_ITEM_COL].unique())
        for user in train[DEFAULT_USER_COL].unique():
            user = [user] * len(item)
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))
        topk_scores = pd.DataFrame(
            data={
                DEFAULT_USER_COL: users,
                DEFAULT_ITEM_COL: items,
                DEFAULT_PREDICTION_COL: preds,
            }
        )
        merged = pd.merge(
            train, topk_scores, on=[DEFAULT_USER_COL, DEFAULT_ITEM_COL], how="outer"
        )
        topk_scores = merged[merged[DEFAULT_RATING_COL].isnull()].drop(
            DEFAULT_RATING_COL, axis=1
        )
    # Remove temp files
    return topk_scores, t


def prepare_training_cornac(train, test):
    return cornac.data.Dataset.from_uir(
        train.drop(DEFAULT_TIMESTAMP_COL, axis=1).itertuples(index=False), seed=SEED
    )


def recommend_k_cornac(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = predict_ranking(
            model,
            train,
            usercol=DEFAULT_USER_COL,
            itemcol=DEFAULT_ITEM_COL,
            predcol=DEFAULT_PREDICTION_COL,
            remove_seen=remove_seen,
        )
    return topk_scores, t


def train_bpr(params, data):
    model = cornac.models.BPR(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def train_bivae(params, data):
    model = cornac.models.BiVAECF(**params)
    with Timer() as t:
        model.fit(data)
    return model, t


def prepare_training_sar(train, test):
    return train


def train_sar(params, data):
    model = SAR(**params)
    model.set_index(data)
    with Timer() as t:
        model.fit(data)
    return model, t


def recommend_k_sar(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = model.recommend_k_items(
            test, top_k=top_k, remove_seen=remove_seen
        )
    return topk_scores, t


def prepare_training_lightgcn(train, test):
    return ImplicitCF(train=train, test=test)


def train_lightgcn(params, data):
    hparams = prepare_hparams(**params)
    model = LightGCN(hparams, data)
    with Timer() as t:
        model.fit()
    return model, t


def recommend_k_lightgcn(model, test, train, top_k=DEFAULT_K, remove_seen=True):
    with Timer() as t:
        topk_scores = model.recommend_k_items(
            test, top_k=top_k, remove_seen=remove_seen
        )
    return topk_scores, t


def rating_metrics_pyspark(test, predictions):
    rating_eval = SparkRatingEvaluation(test, predictions, **COL_DICT)
    evaluation = {
        "RMSE": rating_eval.rmse(),
        "MAE": rating_eval.mae(),
        "MSE": rating_eval.mse(),
        "MAPE": rating_eval.mape(),
        "R2": rating_eval.exp_var(),
        "Explained Variance": rating_eval.rsquared()
    }
    print(evaluation)
    return evaluation


def rating_metrics_python(test, predictions):
    evaluation = {
        "RMSE": rmse(test, predictions, **COL_DICT),
        "MAE": mae(test, predictions, **COL_DICT),
        "MSE": mse(test, predictions, **COL_DICT),
        "MAPE": mape(test, predictions, **COL_DICT),
        "R2": rsquared(test, predictions, **COL_DICT),
        "Explained Variance": exp_var(test, predictions, **COL_DICT)
    }
    print(evaluation)
    return evaluation


def ranking_metrics_pyspark(test, predictions, k=DEFAULT_K):
    rank_eval = SparkRankingEvaluation(
        test, predictions, k=k, relevancy_method="top_k", **COL_DICT
    )
    evaluations = {
        "MAP": rank_eval.map_at_k(),
        "MRR": rank_eval.mrr_at_k(),
        "ARHR@k": rank_eval.arhr_at_k(),
        "Average_Recall@k": rank_eval.average_recall_at_k(),
        "Average_Precision@k": rank_eval.average_precision_at_k(),
        "Precision@k": rank_eval.precision_at_k(),
        "Recall@k": rank_eval.recall_at_k(),
        "nDCG@k": rank_eval.ndcg_at_k(),
    }
    return evaluations


def ranking_metrics_python(test, predictions, k=DEFAULT_K):
    return {
        "MAP": map_at_k(test, predictions, k=k, **COL_DICT),
        "MRR": mrr_at_k(test, predictions, k=k, **COL_DICT),
        "ARHR@k": arhr_at_k(test, predictions, k=k, **COL_DICT),
        "Precision@k": precision_at_k(test, predictions, k=k, **COL_DICT),
        "Recall@k": recall_at_k(test, predictions, k=k, **COL_DICT),
        "Average_Recall@k": average_recall_at_k(test, predictions, k=k, **COL_DICT),
        "Average_Precision@k": average_precision_at_k(test, predictions, k=k, **COL_DICT),
        "nDCG@k": ndcg_at_k(test, predictions, k=k, **COL_DICT)
    }
