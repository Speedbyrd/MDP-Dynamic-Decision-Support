from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.classification import RandomForestClassifier as SparkRF
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
import logging

logger = logging.getLogger(__name__)


def _pandas_to_spark_safe(spark, pdf, chunk_size=50000):
    """Convert pandas DataFrame to Spark in chunks to avoid serialization limit."""
    if pdf.empty:
        return spark.createDataFrame(pdf)
    chunks = [pdf.iloc[i : i + chunk_size] for i in range(0, len(pdf), chunk_size)]
    sdfs = [spark.createDataFrame(chunk) for chunk in chunks]
    result = sdfs[0]
    for sdf in sdfs[1:]:
        result = result.unionByName(sdf, allowMissingColumns=True)
    return result


@configure(profile=["DRIVER_MEMORY_EXTRA_EXTRA_LARGE", "EXECUTOR_MEMORY_LARGE"])
@transform(
    df_initialized=Output(
        "path/to/df_initialized"
    ),
    training=Input("path/to/training"),
    validation=Input("path/to/validation"),
    end_states=Input("path/to/end_states"),
)
def compute(training, validation, end_states, df_initialized, ctx):
    spark = ctx.spark_session

    train_sdf = (
        training.dataframe()
        .where(F.col("ID").isNotNull())
        .withColumn("ID", F.col("ID").cast("string"))
    )
    train_sdf = train_sdf.withColumn(
        "total_sofa",
        F.col("coag_sofa")
        + F.col("cns_sofa")
        + F.col("liver_sofa")
        + F.col("resp_sofa")
        + F.col("renal_sofa")
        + F.col("cardio_sofa"),
    )
    train_sdf = train_sdf.withColumn("RISKwSOFA", F.col("total_sofa") + F.col("RISK"))
    train_cols = train_sdf.columns

    test_sdf = (
        validation.dataframe()
        .where(F.col("ID").isNotNull())
        .withColumn("ID", F.col("ID").cast("string"))
    )
    test_sdf = test_sdf.withColumn(
        "total_sofa",
        F.col("coag_sofa")
        + F.col("cns_sofa")
        + F.col("liver_sofa")
        + F.col("resp_sofa")
        + F.col("renal_sofa")
        + F.col("cardio_sofa"),
    )
    test_sdf = test_sdf.withColumn("RISKwSOFA", F.col("total_sofa") + F.col("RISK"))

    assembler = VectorAssembler(inputCols=["RISKwSOFA"], outputCol="features")
    train_features = assembler.transform(train_sdf).select("features", "RISKwSOFA")

    # wss = []
    # silhouette_scores = []
    best_wss = 1e10
    for k in list(range(2, 6)):
        kmeans = SparkKMeans(
            k=k, seed=0, featuresCol="features", predictionCol="cluster"
        )
        model = kmeans.fit(train_features)
        predictions = model.transform(train_features)
        # WSS (inertia)
        wss = model.summary.trainingCost
        logger.info(f"inertia for {k} clusters: {wss}")
        if wss < best_wss:
            best_wss = wss
            best_predictions = predictions

    # Add cluster assignments to original DataFrame
    train_with_cluster = train_sdf.join(
        best_predictions.select("RISKwSOFA", "cluster"), on="RISKwSOFA", how="left"
    ).withColumn("CLUSTER", F.col("cluster").cast("int"))

    output_cols = train_cols
    existing = [c for c in output_cols if c in train_with_cluster.columns]
    df_initialized.write_dataframe(train_with_cluster.select(existing))