import sys
import pyspark
from pyspark.ml.recommendation import ALS
from pyspark.sql.types import FloatType, IntegerType, LongType, StructType, StructField
import warnings
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.spark_splitters import spark_random_split
from recommenders.evaluation.spark_evaluation import SparkDiversityEvaluation
from recommenders.utils.spark_utils import start_or_get_spark
from pyspark.sql.window import Window
import pyspark.sql.functions as F

warnings.simplefilter(action='ignore', category=FutureWarning)
print("System version: {}".format(sys.version))
print("Spark version: {}".format(pyspark.__version__))

# top k items to recommend
TOP_K = 10
# user, item column names
COL_USER = "UserId"
COL_ITEM = "MovieId"
COL_RATING = "Rating"
COL_TITLE = "Title"
COL_GENRE = "Genre"

# setting up spark
spark = start_or_get_spark("ALS PySpark", memory="16g")
spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", "false")
spark.conf.set("spark.sql.crossJoin.enabled", "true")


def get_diversity_results(diversity_eval):
    metrics = {
        "catalog_coverage": diversity_eval.catalog_coverage(),
        "distributional_coverage": diversity_eval.distributional_coverage(),
        "novelty": diversity_eval.novelty(),
        "diversity": diversity_eval.diversity(),
        "serendipity": diversity_eval.serendipity()
    }
    return metrics


movielens_data_sizes = ["100k", "1m", "10m"]

header = {
    "userCol": COL_USER,
    "itemCol": COL_ITEM,
    "ratingCol": COL_RATING,
}

als = ALS(
    rank=10,
    maxIter=15,
    implicitPrefs=False,
    regParam=0.05,
    coldStartStrategy='drop',
    nonnegative=False,
    seed=42,
    **header
)

for data_size in movielens_data_sizes:
    print(f"Calculating Metrics on Data Size : {data_size}".format(data_size=data_size))
    MOVIELENS_DATA_SIZE = data_size
    schema = StructType(
        (
            StructField(COL_USER, IntegerType()),
            StructField(COL_ITEM, IntegerType()),
            StructField(COL_RATING, FloatType()),
            StructField("Timestamp", LongType()),
        )
    )

    data = movielens.load_spark_df(spark,
                                   size=MOVIELENS_DATA_SIZE,
                                   schema=schema,
                                   title_col=COL_TITLE,
                                   genres_col=COL_GENRE)

    train_df, test_df = spark_random_split(data.select(COL_USER, COL_ITEM, COL_RATING), ratio=0.75, seed=123)
    print("N train_df", train_df.cache().count())
    print("N test_df", test_df.cache().count())

    users = train_df.select(COL_USER).distinct()
    items = train_df.select(COL_ITEM).distinct()
    user_item = users.crossJoin(items)

    with Timer() as train_time:
        model = als.fit(train_df)

    print("Took {} seconds for training.".format(train_time.interval))

    # Score all user-item pairs
    dfs_pred = model.transform(user_item)

    # Remove seen items.
    dfs_pred_exclude_train = dfs_pred.alias("pred").join(
        train_df.alias("train"),
        (dfs_pred[COL_USER] == train_df[COL_USER]) & (dfs_pred[COL_ITEM] == train_df[COL_ITEM]),
        how='outer'
    )

    top_all = dfs_pred_exclude_train.filter(dfs_pred_exclude_train["train.Rating"].isNull()) \
        .select('pred.' + COL_USER, 'pred.' + COL_ITEM, 'pred.' + "prediction")

    window = Window.partitionBy(COL_USER).orderBy(F.col("prediction").desc())
    top_k_reco = top_all.select("*", F.row_number().over(window).alias("rank")).filter(F.col("rank") <= TOP_K).drop("rank")

    als_diversity_eval = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=top_k_reco,
        col_user=COL_USER,
        col_item=COL_ITEM
    )

    als_diversity_metrics = get_diversity_results(als_diversity_eval)
    print(als_diversity_metrics)
    print("*"*10)
