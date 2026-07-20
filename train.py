"""
Fully Spark-native MDP Model Training — runs on ALL patients without OOM.

Performance-tuned version:
- Checkpointing every 3 iterations to truncate Spark DAG lineage
- Single NEXT_CLUSTER recompute per iteration (was 2)
- Lighter RF (20 trees, depth 5) for faster iteration
- Explicit unpersist of old caches
- max_k=15 (clinically sufficient, avoids diminishing returns)
"""

from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.classification import RandomForestClassifier as SparkRF
import math
import logging
from pyspark.sql.types import IntegerType

logger = logging.getLogger(__name__)


def one_step_reward_error(df_trained, P_sdf, R_sdf, end_state_rewards_df):
    # Join transition matrix with reward table to get expected next risk
    P_sdf = P_sdf.withColumnRenamed("NEXT_CLUSTER", "next_cluster")
    R_sdf = R_sdf.withColumnRenamed("CLUSTER", "next_cluster")
    P_w_rewards = P_sdf.join(R_sdf, on="next_cluster", how="left")
    P_w_rewards = P_w_rewards.withColumn(
        "expected_next_risk", F.col("prob") * F.col("RISK")
    )
    # Sum expected_next_risk for each (CLUSTER, ACTION)
    expected_reward_table = P_w_rewards.groupBy("CLUSTER", "ACTION").agg(
        F.sum("expected_next_risk").alias("expected_next_risk")
    )
    # Add expected_next_risk to training data
    df_new = df_trained.join(
        expected_reward_table, on=["CLUSTER", "ACTION"], how="left"
    )
    # Compute NEXT_RISK (actual next risk) using Spark window
    window = Window.partitionBy("ID").orderBy("TIME")
    df_new = df_new.withColumn("NEXT_RISK", F.lead("RISK", 1).over(window))
    # Join end_state_rewards_sdf to get Reward for terminal steps
    end_state_rewards = end_state_rewards_df.select("ID", "Reward")
    df_new = df_new.join(end_state_rewards, on="ID", how="left")
    # Replace null NEXT_RISK with end state Reward
    df_new = df_new.withColumn(
        "NEXT_RISK",
        F.when(F.col("NEXT_RISK").isNull(), F.col("Reward")).otherwise(
            F.col("NEXT_RISK")
        ),
    )
    # Calculate absolute error
    df_new = df_new.withColumn(
        "abs_error", F.abs(F.col("expected_next_risk") - F.col("NEXT_RISK"))
    )
    # Calculate MSE, MAE, RSS, TSS, R2
    mse = df_new.agg(F.mean(F.col("abs_error") ** 2)).collect()[0][0]
    mae = df_new.agg(F.mean(F.col("abs_error"))).collect()[0][0]
    rss = df_new.agg(F.sum(F.col("abs_error") ** 2)).collect()[0][0]
    mean_next_risk = df_new.agg(F.mean(F.col("NEXT_RISK"))).collect()[0][0]
    tss = (
        df_new.withColumn(
            "tss_component", F.abs(mean_next_risk - F.col("NEXT_RISK")) ** 2
        )
        .agg(F.sum("tss_component"))
        .collect()[0][0]
    )
    R2 = 1 - (rss / tss) if tss != 0 else None
    return mae, mse, R2


def compute_most_likely_final_end_state(sdf, P_sdf, end_states_indexed, logger=None):
    """
    For each patient, simulate the MDP using actual actions and always transition to the most likely next cluster.
    Returns accuracy and confusion matrix comparing predicted vs true end state.
    """
    # Step 1: Prepare transition lookup table for (CLUSTER, ACTION) -> most likely NEXT_CLUSTER
    p = P_sdf.alias("p")

    max_probs = (
        P_sdf.groupBy("CLUSTER", "ACTION")
        .agg(F.max("prob").alias("max_prob"))
        .alias("m")
    )

    most_likely_transitions = max_probs.join(
        p,
        (F.col("m.CLUSTER") == F.col("p.CLUSTER"))
        & (F.col("m.ACTION") == F.col("p.ACTION"))
        & (F.col("m.max_prob") == F.col("p.prob")),
        how="inner",
    ).select(
        F.col("p.CLUSTER").alias("CLUSTER"),
        F.col("p.ACTION").alias("ACTION"),
        F.col("p.NEXT_CLUSTER").alias("most_likely_next_cluster"),
    )

    # Step 2: Sort patient trajectories by TIME
    sdf_sorted = sdf.orderBy("ID", "TIME")

    # Step 3: Collect actions and clusters for each patient
    patient_paths = sdf_sorted.groupBy("ID").agg(
        F.collect_list("CLUSTER").alias("cluster_seq"),
        F.collect_list("ACTION").alias("action_seq"),
    )

    # Step 4: Define UDF to simulate path through MDP
    def simulate_path(cluster_seq, action_seq, transition_dict):
        current_cluster = cluster_seq[0]
        for action in action_seq:
            key = (current_cluster, action)
            if key in transition_dict:
                current_cluster = transition_dict[key]
            # else, stay in current_cluster
        return current_cluster

    # Build transition_dict in driver
    transition_rows = most_likely_transitions.collect()
    transition_dict = {
        (int(row["CLUSTER"]), int(row["ACTION"])): int(row["most_likely_next_cluster"])
        for row in transition_rows
    }

    from pyspark.sql.types import ArrayType
    from pyspark.sql.functions import udf

    simulate_udf = udf(
        lambda clusters, actions: simulate_path(clusters, actions, transition_dict),
        IntegerType(),
    )

    # Step 5: Apply UDF to get predicted final cluster for each patient
    patient_paths = patient_paths.withColumn(
        "predicted_final_cluster", simulate_udf("cluster_seq", "action_seq")
    )

    # Step 6: Join with true end state
    patient_paths = patient_paths.join(
        end_states_indexed.select("ID", "end_state_indexed"), on="ID", how="left"
    )

    # Step 7: Compute accuracy and confusion matrix
    patient_paths = patient_paths.withColumn(
        "correct", F.col("predicted_final_cluster") == F.col("end_state_indexed")
    )
    accuracy = patient_paths.agg(F.mean(F.col("correct").cast("double"))).collect()[0][
        0
    ]
    confusion_df = (
        patient_paths.groupBy("predicted_final_cluster", "end_state_indexed")
        .agg(F.count("*").alias("count"))
        .orderBy("predicted_final_cluster", "end_state_indexed")
    )

    if logger:
        logger.info(
            f"(Split) MDP most likely end state prediction accuracy: {accuracy:.4f}"
        )
        logger.info("(Split) MDP most likely end state confusion matrix:")
        for row in confusion_df.collect():
            logger.info(
                f"(Split) Predicted: {row['predicted_final_cluster']}, True: {row['end_state_indexed']}, Count: {row['count']}"
            )

    return accuracy, confusion_df


def _recompute_next_cluster(sdf, end_states, spark):
    # number the clusters
    """Compute NEXT_CLUSTER using window lead, filling last timestep with ID-specific end state."""
    max_cluster = sdf.agg(F.max("CLUSTER")).collect()[0][0]
    offset = max_cluster + 1
    # Get unique end states
    unique_end_states = [
        row["end_state"] for row in end_states.select("end_state").distinct().collect()
    ]
    # Map each end state to an integer starting from offset
    end_state_mapping = {
        state: idx + offset for idx, state in enumerate(unique_end_states)
    }
    # Create a mapping DataFrame
    mapping_df = spark.createDataFrame(
        [(state, idx) for state, idx in end_state_mapping.items()],
        ["end_state", "end_state_indexed"],
    )
    # join mapping to end state df
    end_states_indexed = end_states.join(mapping_df, on="end_state", how="left")

    w = Window.partitionBy("ID").orderBy("TIME")
    sdf = sdf.drop("NEXT_CLUSTER") if "NEXT_CLUSTER" in sdf.columns else sdf
    # adding end state column
    sdf_with_end_state = sdf.join(
        end_states_indexed.select("ID", F.col("end_state_indexed").alias("END_STATE")),
        on="ID",
        how="left",
    )

    return sdf_with_end_state.withColumn(
        "NEXT_CLUSTER", F.coalesce(F.lead("CLUSTER").over(w), F.col("END_STATE"))
    ).drop("END_STATE")


def _compute_incoherence(sdf, min_obs=500):
    logger.info("(Split) deterministic count incoherence metric")
    """Incoherence per (cluster, action) = stddev of NEXT_CLUSTER within group."""
    return (
        sdf.where(F.col("NEXT_CLUSTER").isNotNull())
        .groupBy("CLUSTER", "ACTION")
        .agg(
            F.stddev("NEXT_CLUSTER").alias("incoherence"),
            F.count("*").alias("cnt"),
        )
        .where((F.col("cnt") >= min_obs) & F.col("incoherence").isNotNull())
        .orderBy(F.col("incoherence").desc())
    )


def _compute_information_radius(sdf, min_obs=1000):
    logger.info("(Split) JSD incoherence metric")
    prob_cols = [col for col in sdf.columns if col.startswith("prob_next_cluster_")]
    grouped = (
        sdf.where(F.col("NEXT_CLUSTER").isNotNull())
        .groupBy("CLUSTER", "ACTION")
        .agg(
            F.collect_list(F.array(*prob_cols)).alias("prob_vectors"),
            F.count("*").alias("cnt"),
        )
        # .where(F.col("cnt") >= min_obs)
    )

    def jsd(p, q):
        # Both p and q are lists of floats, sum to 1
        m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]

        def kl(x, y):
            s = 0.0
            for xi, yi in zip(x, y):
                if xi > 0 and yi > 0:
                    s += xi * math.log(xi / yi)
            return s

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    def info_radius_udf(prob_vectors):
        X = [list(map(float, row)) for row in prob_vectors]
        n = len(X)
        if n == 0:
            return float("nan")
        # Compute mean vector
        d = len(X[0])
        avg = [0.0] * d
        for row in X:
            for i in range(d):
                avg[i] += row[i]
        avg = [x / n for x in avg]
        # Compute mean JSD
        jsd_sum = 0.0
        for row in X:
            jsd_sum += jsd(row, avg)
        return jsd_sum / n

    from pyspark.sql.types import DoubleType

    info_radius = F.udf(info_radius_udf, DoubleType())
    result = grouped.withColumn("incoherence", info_radius(F.col("prob_vectors")))
    return result.select("CLUSTER", "ACTION", "incoherence", "cnt").orderBy(
        F.col("incoherence").desc()
    )


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


def split_worst_cluster_with_rf_and_kmeans(sdf, worst, prob_cols, logger=None):
    """
    Splits the worst cluster/action using K-means and classifies remaining points with Random Forest.

    Args:
        sdf (DataFrame): Input Spark DataFrame.
        worst (dict): Dict with keys 'CLUSTER' and 'ACTION' indicating the worst cluster/action.
        prob_cols (list): List of column names for transition probabilities.
        logger (optional): Logger object for info/debug output.

    Returns:
        DataFrame: DataFrame with 'final_sub_cluster' column, combining K-means and RF results.
    """
    # Prepare features for K-means: transition probabilities + NEXT_CLUSTER
    kmeans_features = prob_cols + ["NEXT_CLUSTER"]
    split_asm = VectorAssembler(
        inputCols=kmeans_features, outputCol="transition_vec", handleInvalid="skip"
    )
    # Subset with worst cluster AND worst action
    kmeans_subset = sdf.where(
        (F.col("CLUSTER") == int(worst["CLUSTER"]))
        & (F.col("ACTION") == int(worst["ACTION"]))
    )
    kmeans_subset_vec = split_asm.transform(kmeans_subset)
    # K-means clustering (k=2)
    split_km = SparkKMeans(
        k=2, seed=0, featuresCol="transition_vec", predictionCol="kmeans_label"
    )
    kmeans_model = split_km.fit(kmeans_subset_vec)
    kmeans_labeled = kmeans_model.transform(kmeans_subset_vec)

    # Prepare remaining points in worst cluster (different actions)
    remaining_worst_cluster = sdf.where(
        (F.col("CLUSTER") == int(worst["CLUSTER"]))
        & (F.col("ACTION") != int(worst["ACTION"]))
    )

    if remaining_worst_cluster.count() > 0:
        # Train Random Forest using ALL FEATURES except excluded columns
        exclude_cols = {
            "ID",
            "CLUSTER",
            "NEXT_CLUSTER",
            "transition_vec",
            "kmeans_label",
            "rf_features",
            "rf_prediction",
            "rf_label",
            "final_sub_cluster",
        }
        all_feature_cols = [col for col in sdf.columns if col not in exclude_cols]
        rf_asm = VectorAssembler(
            inputCols=all_feature_cols,
            outputCol="rf_features",
            handleInvalid="skip",
        )
        # Prepare training data (K-means labeled subset) with all features
        rf_train_data = rf_asm.transform(kmeans_labeled)
        rf_train_data = rf_train_data.withColumn(
            "rf_label", F.col("kmeans_label").cast("double")
        )
        # Train Random Forest classifier
        rf_classifier = SparkRF(
            featuresCol="rf_features",
            labelCol="rf_label",
            numTrees=20,
            maxDepth=7,
            seed=0,
            predictionCol="rf_prediction",
        )
        rf_model = rf_classifier.fit(rf_train_data.where(F.col("rf_label").isNotNull()))
        # Classify remaining points in worst cluster
        remaining_vec = rf_asm.transform(remaining_worst_cluster)
        remaining_classified = rf_model.transform(remaining_vec)
        # Combine K-means and RF results
        kmeans_final = kmeans_labeled.withColumn(
            "final_sub_cluster", F.col("kmeans_label")
        )
        rf_final = remaining_classified.withColumn(
            "final_sub_cluster", F.col("rf_prediction").cast("integer")
        )
        # Select consistent columns for union
        common_cols = [
            col
            for col in sdf.columns
            if col in kmeans_final.columns and col in rf_final.columns
        ]
        common_cols.append("final_sub_cluster")
        all_worst_cluster_split = kmeans_final.select(common_cols).unionByName(
            rf_final.select(common_cols), allowMissingColumns=True
        )
        # if logger:
        #     logger.info(
        #         f"(Split) RF classified {remaining_classified.count()} additional points using all features"
        #     )
    else:
        # Only K-means results if no other actions in worst cluster
        all_worst_cluster_split = kmeans_labeled.withColumn(
            "final_sub_cluster", F.col("kmeans_label")
        )
    return all_worst_cluster_split


@configure(profile=["DRIVER_MEMORY_EXTRA_EXTRA_LARGE", "EXECUTOR_MEMORY_LARGE"])
@transform(
    df_trained=Output("/path/to/df_trained"),
    R_df=Output("/path/to/R_df"),
    P_df=Output("/path/to/P_df"),
    features=Input("/path/to/features"),
    end_states=Input("/path/to/end_states"),
)
def compute(features, end_states, df_trained_out, R_df_out, P_df_out, ctx):
    spark = ctx.spark_session

    # ── Config ────────────────────────────────────────────────────────────
    # pfeatures = 190
    # n_clusters_init = 4
    max_k = 8  # 15 clusters is clinically sufficient; avoids 28 slow iterations
    min_obs = 5000  # smallest a cluster can be to still be considered for splitting
    min_splitoff = 300 # the smallest number of points that can be split off into a new cluster
    RF_NUM_TREES = 20  # Lighter RF per iteration (still effective for cluster assignment)
    RF_MAX_DEPTH = 5
    CHECKPOINT_EVERY = 3  # Truncate Spark DAG lineage every N iterations
    # ── 1. Load data (stays distributed) ──────────────────────────────────
    
    end_states_df = end_states.dataframe()
    sdf = features.dataframe().where(F.col("ID").isNotNull()).where(F.col("TIME") >= 0)
    sdf = sdf.withColumn("ID", F.col("ID").cast("string"))
    sdf = sdf.withColumn("RISK", F.col("total_sofa") + F.col("RISK"))

    all_cols = sdf.columns
    feature_cols = [
        col
        for col in all_cols
        if col not in ["ID", "TIME", "ACTION", "RISK", "CLUSTER", "NEXT_CLUSTER"]
    ]

    # ── 2. Initial clustering by RISK (Spark KMeans) ──────────────────────
    if "CLUSTER" not in all_cols:
        n_clusters_init = 3
        risk_asm = VectorAssembler(
            inputCols=["RISK"], outputCol="risk_vec", handleInvalid="skip"
        )
        sdf_risk = risk_asm.transform(sdf)
        km = SparkKMeans(
            k=n_clusters_init, seed=0, featuresCol="risk_vec", predictionCol="CLUSTER"
        )
        sdf = km.fit(sdf_risk).transform(sdf_risk).drop("risk_vec")
        sdf = sdf.withColumn("CLUSTER", F.col("CLUSTER").cast("integer"))
    else:
        logger.info("(Split) pre-initialized clusters")
        n_clusters_init = sdf.select("CLUSTER").distinct().count()

    # Compute NEXT_CLUSTER
    sdf = _recompute_next_cluster(sdf, end_states_df, spark)
    sdf = sdf.localCheckpoint(eager=True)  # materialize + truncate lineage
    # sdf = sdf.persist()
    total_rows = sdf.count()
    n_clusters = n_clusters_init
    # logger.info(f"Rows: {total_rows:,} | Init clusters: {n_clusters_init} | Target: {max_k}")

    prev_sdf = None  # track for unpersist
    # ── 3. Iterative splitting loop (all Spark, no toPandas) ──────────────
    tried = []  # (c,a) pairs that we've already tried to split
    for iteration in range(max_k - n_clusters_init):
        sdf_cluster_value_counts = sdf.groupBy("CLUSTER").count()
        rows = sdf_cluster_value_counts.collect()
        for row in rows:
            logger.info(f"(Split) Cluster: {row['CLUSTER']}, Count: {row['count']}")

        # Prepare features for RF training
        rf_feature_cols = feature_cols + ["ACTION"]  # Include ACTION as a feature
        rf_asm = VectorAssembler(
            inputCols=rf_feature_cols, outputCol="rf_feat", handleInvalid="skip"
        )
        sdf_rf = rf_asm.transform(sdf)

        # Prepare labels - use NEXT_CLUSTER as target
        sdf_rf = sdf_rf.withColumn(
            "NEXT_CLUSTER_label", F.col("NEXT_CLUSTER").cast("double")
        )

        # Train Random Forest classifier
        rf = SparkRF(
            featuresCol="rf_feat",
            labelCol="NEXT_CLUSTER_label",
            numTrees=RF_NUM_TREES,
            maxDepth=RF_MAX_DEPTH,
            seed=0,
            probabilityCol="rf_probabilities",  # Ensure probability output
        )

        # Fit model only on rows with non-null NEXT_CLUSTER
        rf_model = rf.fit(sdf_rf.where(F.col("NEXT_CLUSTER_label").isNotNull()))

        # Get predictions with probabilities
        preds = rf_model.transform(sdf_rf)

        # Extract probability columns for each possible NEXT_CLUSTER value
        # Get unique NEXT_CLUSTER values to create probability columns
        unique_clusters = (
            sdf.select("NEXT_CLUSTER")
            .where(F.col("NEXT_CLUSTER").isNotNull())
            .distinct()
            .rdd.map(lambda row: row[0])
            .collect()
        )
        unique_clusters = sorted([int(c) for c in unique_clusters if c is not None])

        # logger.info(f"Creating probability columns for {len(unique_clusters)} clusters: {unique_clusters}")

        # Add probability columns for each cluster
        from pyspark.ml.linalg import VectorUDT
        from pyspark.sql.types import DoubleType

        # Function to extract probability for specific cluster index
        def extract_prob(cluster_idx):
            def _extract(prob_vector):
                if prob_vector is not None and len(prob_vector) > cluster_idx:
                    return float(prob_vector[cluster_idx])
                return 0.0

            return F.udf(_extract, DoubleType())

        # Add probability columns for each unique cluster
        for i, cluster_id in enumerate(unique_clusters):
            prob_col_name = f"prob_next_cluster_{cluster_id}"
            preds = preds.withColumn(
                prob_col_name, extract_prob(i)(F.col("rf_probabilities"))
            )

        # Update sdf with the probability columns
        # prob_cols = [f"prob_next_cluster_{c}" for c in unique_clusters]
        columns_to_keep = [
            col
            for col in preds.columns
            if col
            not in [
                "rf_feat",
                "NEXT_CLUSTER_label",
                "rawPrediction",
                "rf_probabilities",
                "prediction",
            ]
        ]
        prob_cols = [col for col in sdf.columns if col.startswith("prob_next_cluster_")]
        sdf = preds.select(columns_to_keep)
        sdf = sdf.localCheckpoint(eager=True)

        # Evaluate incoherence
        incoh = _compute_information_radius(sdf, min_obs)
        split_success = False
        # tried = []
        for row in incoh.collect():
            if (row["CLUSTER"], row["ACTION"]) not in tried and row["cnt"] >= min_obs:
                logger.info(
                    f"Split try cluster: {row['CLUSTER']}, action: {row['ACTION']}, incoherence={row['incoherence']}, n={row['cnt']}"
                )
                tried.append((row["CLUSTER"], row["ACTION"]))
                all_worst_cluster_split = split_worst_cluster_with_rf_and_kmeans(
                    sdf, row, prob_cols, logger
                )
                subcluster1 = (
                    all_worst_cluster_split.where(F.col("final_sub_cluster") == 1)
                    .withColumn("CLUSTER", F.col("final_sub_cluster"))
                    .drop(
                        "transition_vec",
                        "rf_features",
                        "kmeans_label",
                        "rf_prediction",
                        "rf_label",
                        "final_sub_cluster",
                    )
                )
                subcluster1_size = subcluster1.count()
                # worst_incoherence1 = _compute_information_radius(subcluster1, min_obs).collect()[0]['incoherence']
                subcluster0 = (
                    all_worst_cluster_split.where(F.col("final_sub_cluster") == 0)
                    .withColumn("CLUSTER", F.col("final_sub_cluster"))
                    .drop(
                        "transition_vec",
                        "rf_features",
                        "kmeans_label",
                        "rf_prediction",
                        "rf_label",
                        "final_sub_cluster",
                    )
                )
                subcluster0_size = subcluster0.count()
                # worst_incoherence0 = _compute_information_radius(subcluster0, min_obs).collect()[0]['incoherence']
                if subcluster0_size >= min_splitoff and subcluster1_size >= min_splitoff:
                    worst_incoherence0 = _compute_information_radius(
                        subcluster0, min_splitoff
                    ).collect()[0]["incoherence"]
                    worst_incoherence1 = _compute_information_radius(
                        subcluster1, min_splitoff
                    ).collect()[0]["incoherence"]
                    logger.info(
                        f"(Split) subcluster 0 incoherence: {worst_incoherence0}, size: {subcluster0_size}"
                    )
                    logger.info(
                        f"(Split) subcluster 1 incoherence: {worst_incoherence1}, size: {subcluster1_size}"
                    )
                    # if (
                    #     worst_incoherence0 < 2 * row["incoherence"]
                    #     and worst_incoherence1 < 2 * row["incoherence"]
                    # ):
                    split_success = True
                    split_row = row
                    split_cluster = row["CLUSTER"]
                    break
                else:
                    logger.info(f"(Split) subcluster 0 size: {subcluster0_size}")
                    logger.info(f"(Split) subcluster 1 size: {subcluster1_size}")

        if split_success == True:
            # Apply cluster renumbering based on final sub-cluster assignments
            new_cluster_number = n_clusters
            subset_new = all_worst_cluster_split.withColumn(
                "CLUSTER",
                F.when(F.col("final_sub_cluster") == 0, F.col("CLUSTER")).otherwise(
                    F.lit(new_cluster_number).cast("integer")
                ),
            ).drop(
                "transition_vec",
                "rf_features",
                "kmeans_label",
                "rf_prediction",
                "rf_label",
                "final_sub_cluster",
            )
            # Get data points not in the worst cluster (unchanged)
            rest = sdf.where(F.col("CLUSTER") != int(split_row["CLUSTER"]))
            logger.info(
                f"Split COMPLETE: {subset_new.where(F.col('CLUSTER') == new_cluster_number).count()} points moved to new cluster {new_cluster_number} from cluster {split_cluster}"
            )
            # Track old for unpersist
            prev_sdf = sdf
            # Union and recompute NEXT_CLUSTER (single recompute per iteration)
            sdf = rest.unionByName(
                subset_new, allowMissingColumns=True
            )  # what is unionByName
            sdf = _recompute_next_cluster(sdf, end_states_df, spark)
            n_clusters = n_clusters + 1
            # calculating the new MDP and predictive metrics
            max_cluster = sdf.agg(F.max("CLUSTER")).collect()[0][0]
            offset = max_cluster + 1
            # Get unique end states
            unique_end_states = [
                row["end_state"]
                for row in end_states_df.select("end_state").distinct().collect()
            ]
            # Map each end state to an integer starting from offset
            end_state_mapping = {
                state: idx + offset for idx, state in enumerate(unique_end_states)
            }
            # Create a mapping DataFrame
            mapping_df = spark.createDataFrame(
                [(state, idx) for state, idx in end_state_mapping.items()],
                ["end_state", "end_state_indexed"],
            )
            # join mapping to end state df
            end_states_indexed = end_states_df.join(
                mapping_df, on="end_state", how="left"
            )
            # ── 5. Compute P and R matrices (tiny aggregations) ───────────────────
            # logger.info("Computing transition probabilities (P) and rewards (R)...")
            P_sdf = (
                sdf.where(F.col("NEXT_CLUSTER").isNotNull())
                .groupBy("CLUSTER", "ACTION", "NEXT_CLUSTER")
                .agg(F.count("*").alias("cnt"))
            )
            totals = P_sdf.groupBy("CLUSTER", "ACTION").agg(F.sum("cnt").alias("total"))
            P_sdf = (
                P_sdf.join(totals, on=["CLUSTER", "ACTION"])
                .withColumn("prob", F.col("cnt") / F.col("total"))
                .select("CLUSTER", "ACTION", "NEXT_CLUSTER", "prob")
            )
            # Existing R_sdf for regular clusters (remove cnt column)
            R_sdf = (
                sdf.groupBy("CLUSTER")
                .agg(F.mean("RISK").alias("RISK"))
                .select("CLUSTER", "RISK")
            )
            # Prepare end states as additional clusters (one row per end state, no cnt)
            end_states_clusters = end_states_indexed.select(
                F.col("end_state_indexed").alias("CLUSTER"),
                F.col("Reward").alias("RISK"),
            ).distinct()  # Ensures one row per end state
            # Union the two DataFrames (CLUSTER, RISK only)
            R_sdf = R_sdf.unionByName(end_states_clusters, allowMissingColumns=True)
            # accuracy, confusion_df = compute_most_likely_final_end_state(
            #     sdf, P_sdf, end_states_indexed, logger
            # )
            mae, mse, R2 = one_step_reward_error(sdf, P_sdf, R_sdf, end_states_indexed)
            logger.info(
                f"MDP one step reward prediction error after Split: MAE={mae:.4f}, MSE={mse:.4f}, R2={R2:.4f}"
            )
        else:
            logger.info("Split failed")
            break

        # Checkpoint every N iterations to truncate lineage
        if (iteration + 1) % CHECKPOINT_EVERY == 0:
            # logger.info(f"Checkpointing (lineage truncation)...")
            sdf = sdf.localCheckpoint(eager=True)
        else:
            sdf = sdf.cache()
            sdf.count()  # materialize

        # Unpersist old DataFrame
        if prev_sdf is not None:
            try:
                prev_sdf.unpersist()
            except Exception:
                pass

        # logger.info(f"(Split) end of iteration number of clusters: {n_clusters + 1}")

    sdf = sdf.localCheckpoint(eager=True)
    max_cluster = sdf.agg(F.max("CLUSTER")).collect()[0][0]
    offset = max_cluster + 1
    # Get unique end states
    unique_end_states = [
        row["end_state"]
        for row in end_states_df.select("end_state").distinct().collect()
    ]
    # Map each end state to an integer starting from offset
    end_state_mapping = {
        state: idx + offset for idx, state in enumerate(unique_end_states)
    }
    # Create a mapping DataFrame
    mapping_df = spark.createDataFrame(
        [(state, idx) for state, idx in end_state_mapping.items()],
        ["end_state", "end_state_indexed"],
    )
    # join mapping to end state df
    end_states_indexed = end_states_df.join(mapping_df, on="end_state", how="left")

    # ── 5. Compute P and R matrices (tiny aggregations) ───────────────────
    # logger.info("Computing transition probabilities (P) and rewards (R)...")
    P_sdf = (
        sdf.where(F.col("NEXT_CLUSTER").isNotNull())
        .groupBy("CLUSTER", "ACTION", "NEXT_CLUSTER")
        .agg(F.count("*").alias("cnt"))
    )
    totals = P_sdf.groupBy("CLUSTER", "ACTION").agg(F.sum("cnt").alias("total"))
    P_sdf = (
        P_sdf.join(totals, on=["CLUSTER", "ACTION"])
        .withColumn("prob", F.col("cnt") / F.col("total"))
        .select("CLUSTER", "ACTION", "NEXT_CLUSTER", "prob")
    )
    # Existing R_sdf for regular clusters (remove cnt column)
    R_sdf = (
        sdf.groupBy("CLUSTER")
        .agg(F.mean("RISK").alias("RISK"))
        .select("CLUSTER", "RISK")
    )

    # Prepare end states as additional clusters (one row per end state, no cnt)
    end_states_clusters = end_states_indexed.select(
        F.col("end_state_indexed").alias("CLUSTER"), F.col("Reward").alias("RISK")
    ).distinct()  # Ensures one row per end state

    # Union the two DataFrames (CLUSTER, RISK only)
    R_sdf = R_sdf.unionByName(end_states_clusters, allowMissingColumns=True)

    P_pdf = P_sdf.toPandas()
    R_pdf = R_sdf.toPandas()
    # logger.info(f"P matrix: {len(P_pdf)} entries | R matrix: {len(R_pdf)} clusters")

    # ── 6. Write outputs ──────────────────────────────────────────────────
    output_cols = (
        ["ID", "TIME"] + feature_cols + ["ACTION", "RISK", "CLUSTER", "NEXT_CLUSTER"]
    )
    existing = [c for c in output_cols if c in sdf.columns]
    df_trained_out.write_dataframe(sdf.select(existing))
    R_df_out.write_dataframe(_pandas_to_spark_safe(spark, R_pdf))
    P_df_out.write_dataframe(_pandas_to_spark_safe(spark, P_pdf))

    # logger.info("All outputs written successfully.")