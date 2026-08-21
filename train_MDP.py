from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.classification import RandomForestClassifier as SparkRF
import math
import logging
from pyspark.sql.types import IntegerType
import numpy as np




# ==========================
# CLUSTERING
# ==========================

def train_clusters(features, end_states):
    # Config
    max_k = 8
    min_obs = 5000
    min_splitoff = 300

    RF_NUM_TREES = 20
    RF_MAX_DEPTH = 5
    CHECKPOINT_EVERY = 3

    # 1. Load data
    end_states_df = end_states.dataframe()

    sdf = (
        features.dataframe()
        .where(F.col("ID").isNotNull())
        .where(F.col("TIME") >= 0)
        .withColumn("ID", F.col("ID").cast("string"))
        .withColumn("RISK", F.col("total_sofa") + F.col("RISK"))
    )

    all_cols = sdf.columns

    feature_cols = [
        col
        for col in all_cols
        if col not in [
            "ID",
            "TIME",
            "ACTION",
            "RISK",
            "CLUSTER",
            "NEXT_CLUSTER",
        ]
    ]

    # 2. Initial clustering by RISK
    if "CLUSTER" not in all_cols:
        n_clusters_init = 3

        risk_asm = VectorAssembler(
            inputCols=["RISK"],
            outputCol="risk_vec",
            handleInvalid="skip",
        )

        sdf_risk = risk_asm.transform(sdf)

        km = SparkKMeans(
            k=n_clusters_init,
            seed=0,
            featuresCol="risk_vec",
            predictionCol="CLUSTER",
        )

        sdf = (
            km.fit(sdf_risk)
            .transform(sdf_risk)
            .drop("risk_vec")
            .withColumn("CLUSTER", F.col("CLUSTER").cast("integer"))
        )

    else:
        print("(Split) pre-initialized clusters")

        n_clusters_init = (
            sdf
            .select("CLUSTER")
            .distinct()
            .count()
        )

    sdf = _recompute_next_cluster(sdf, end_states_df)

    sdf = sdf.localCheckpoint(eager=True)

    n_clusters = n_clusters_init
    prev_sdf = None

    # 3. Iterative splitting 
    tried = []

    for iteration in range(max_k - n_clusters_init):

        # Log cluster sizes
        cluster_counts = (
            sdf
            .groupBy("CLUSTER")
            .count()
            .collect()
        )

        for row in cluster_counts:
            print(
                f"(Split) Cluster: {row['CLUSTER']}, "
                f"Count: {row['count']}"
            )

        # Random forest training data 
        rf_feature_cols = feature_cols + ["ACTION"]

        rf_asm = VectorAssembler(
            inputCols=rf_feature_cols,
            outputCol="rf_feat",
            handleInvalid="skip",
        )

        sdf_rf = (
            rf_asm.transform(sdf)
            .withColumn(
                "NEXT_CLUSTER_label",
                F.col("NEXT_CLUSTER").cast("double"),
            )
        )

        # Random Forest
        rf = SparkRF(
            featuresCol="rf_feat",
            labelCol="NEXT_CLUSTER_label",
            numTrees=RF_NUM_TREES,
            maxDepth=RF_MAX_DEPTH,
            seed=0,
            probabilityCol="rf_probabilities",
        )

        rf_model = rf.fit(
            sdf_rf.where(
                F.col("NEXT_CLUSTER_label").isNotNull()
            )
        )

        preds = rf_model.transform(sdf_rf)

        # Find possible NEXT_CLUSTER values 
        unique_clusters = (
            sdf
            .select("NEXT_CLUSTER")
            .where(F.col("NEXT_CLUSTER").isNotNull())
            .distinct()
            .rdd
            .map(lambda row: row[0])
            .collect()
        )

        unique_clusters = sorted(
            int(c)
            for c in unique_clusters
            if c is not None
        )

        # Extract RF probability columns
        def extract_prob(cluster_idx):

            def _extract(prob_vector):
                if (
                    prob_vector is not None
                    and len(prob_vector) > cluster_idx
                ):
                    return float(prob_vector[cluster_idx])

                return 0.0

            return F.udf(_extract, DoubleType())

        for i, cluster_id in enumerate(unique_clusters):
            prob_col_name = (
                f"prob_next_cluster_{cluster_id}"
            )

            preds = preds.withColumn(
                prob_col_name,
                extract_prob(i)(
                    F.col("rf_probabilities")
                ),
            )

        columns_to_keep = [
            col
            for col in preds.columns
            if col not in [
                "rf_feat",
                "NEXT_CLUSTER_label",
                "rawPrediction",
                "rf_probabilities",
                "prediction",
            ]
        ]

        sdf = preds.select(columns_to_keep)
        sdf = sdf.localCheckpoint(eager=True)

        # Important:
        # In your original code this was based on the OLD sdf columns,
        # which means it could accidentally be empty.
        prob_cols = [
            col
            for col in sdf.columns
            if col.startswith("prob_next_cluster_")
        ]

        # Evaluate incoherence 
        incoh = _compute_information_radius(
            sdf,
            min_obs,
        )

        split_success = False

        for row in incoh.collect():

            cluster_action = (
                row["CLUSTER"],
                row["ACTION"],
            )

            if (
                cluster_action not in tried
                and row["cnt"] >= min_obs
            ):
                print(
                    f"Split try cluster: {row['CLUSTER']}, "
                    f"action: {row['ACTION']}, "
                    f"incoherence={row['incoherence']}, "
                    f"n={row['cnt']}"
                )

                tried.append(cluster_action)

                all_worst_cluster_split = (
                    split_worst_cluster_with_rf_and_kmeans(
                        sdf,
                        row,
                        prob_cols,
                    )
                )

                # Candidate subcluster 1
                subcluster1 = (
                    all_worst_cluster_split
                    .where(
                        F.col("final_sub_cluster") == 1
                    )
                    .withColumn(
                        "CLUSTER",
                        F.col("final_sub_cluster"),
                    )
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

                # Candidate subcluster 0
                subcluster0 = (
                    all_worst_cluster_split
                    .where(
                        F.col("final_sub_cluster") == 0
                    )
                    .withColumn(
                        "CLUSTER",
                        F.col("final_sub_cluster"),
                    )
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

                # Check whether split is large enough
                if (
                    subcluster0_size >= min_splitoff
                    and subcluster1_size >= min_splitoff
                ):
                    worst_incoherence0 = (
                        _compute_information_radius(
                            subcluster0,
                            min_splitoff,
                        )
                        .collect()[0]["incoherence"]
                    )

                    worst_incoherence1 = (
                        _compute_information_radius(
                            subcluster1,
                            min_splitoff,
                        )
                        .collect()[0]["incoherence"]
                    )

                    print(
                        "(Split) subcluster 0 "
                        f"incoherence: {worst_incoherence0}, "
                        f"size: {subcluster0_size}"
                    )

                    print(
                        "(Split) subcluster 1 "
                        f"incoherence: {worst_incoherence1}, "
                        f"size: {subcluster1_size}"
                    )

                    split_success = True
                    split_row = row
                    split_cluster = row["CLUSTER"]

                    break

                else:
                    print(
                        "(Split) subcluster 0 "
                        f"size: {subcluster0_size}"
                    )
                    print(
                        "(Split) subcluster 1 "
                        f"size: {subcluster1_size}"
                    )

        # Apply successful split
        if split_success:

            new_cluster_number = n_clusters

            subset_new = (
                all_worst_cluster_split
                .withColumn(
                    "CLUSTER",
                    F.when(
                        F.col("final_sub_cluster") == 0,
                        F.col("CLUSTER"),
                    ).otherwise(
                        F.lit(
                            new_cluster_number
                        ).cast("integer")
                    ),
                )
                .drop(
                    "transition_vec",
                    "rf_features",
                    "kmeans_label",
                    "rf_prediction",
                    "rf_label",
                    "final_sub_cluster",
                )
            )

            rest = sdf.where(
                F.col("CLUSTER")
                != int(split_row["CLUSTER"])
            )

            print(
                "Split COMPLETE: "
                f"{subset_new.where(F.col('CLUSTER') == new_cluster_number).count()} "
                f"points moved to new cluster "
                f"{new_cluster_number} "
                f"from cluster {split_cluster}"
            )

            prev_sdf = sdf

            sdf = rest.unionByName(
                subset_new,
                allowMissingColumns=True,
            )

            sdf = _recompute_next_cluster(
                sdf,
                end_states_df,
            )

            n_clusters += 1

            # Evaluate resulting MDP
            max_cluster = (
                sdf
                .agg(F.max("CLUSTER"))
                .collect()[0][0]
            )

            offset = max_cluster + 1

            # Build indexed end states
            end_state_window = Window.orderBy(
                F.col("end_state")
            )

            end_states_indexed = (
                end_states_df
                .withColumn(
                    "end_state_indexed",
                    (
                        F.dense_rank()
                        .over(end_state_window)
                        - 1
                        + F.lit(offset)
                    ).cast("integer"),
                )
            )

            # Transition matrix P
            P_sdf = (
                sdf
                .where(
                    F.col("NEXT_CLUSTER").isNotNull()
                )
                .groupBy(
                    "CLUSTER",
                    "ACTION",
                    "NEXT_CLUSTER",
                )
                .agg(
                    F.count("*").alias("cnt")
                )
            )

            totals = (
                P_sdf
                .groupBy("CLUSTER", "ACTION")
                .agg(
                    F.sum("cnt").alias("total")
                )
            )

            P_sdf = (
                P_sdf
                .join(
                    totals,
                    on=["CLUSTER", "ACTION"],
                )
                .withColumn(
                    "prob",
                    F.col("cnt") / F.col("total"),
                )
                .select(
                    "CLUSTER",
                    "ACTION",
                    "NEXT_CLUSTER",
                    "prob",
                )
            )

            # Reward matrix R
            R_sdf = (
                sdf
                .groupBy("CLUSTER")
                .agg(
                    F.mean("RISK").alias("RISK")
                )
                .select(
                    "CLUSTER",
                    "RISK",
                )
            )

            end_states_clusters = (
                end_states_indexed
                .select(
                    F.col(
                        "end_state_indexed"
                    ).alias("CLUSTER"),
                    F.col("Reward").alias("RISK"),
                )
                .distinct()
            )

            R_sdf = R_sdf.unionByName(
                end_states_clusters,
                allowMissingColumns=True,
            )

            mae, mse, R2 = one_step_reward_error(
                sdf,
                P_sdf,
                R_sdf,
                end_states_indexed,
            )

            print(
                "MDP one step reward prediction "
                "error after Split: "
                f"MAE={mae:.4f}, "
                f"MSE={mse:.4f}, "
                f"R2={R2:.4f}"
            )

        else:
            print("Split failed")
            break

        # Lineage management
        if (
            iteration + 1
        ) % CHECKPOINT_EVERY == 0:

            sdf = sdf.localCheckpoint(
                eager=True
            )

        else:
            sdf = sdf.cache()
            sdf.count()

        if prev_sdf is not None:
            try:
                prev_sdf.unpersist()
            except Exception:
                pass

    # 4. Finalize 
    sdf = sdf.localCheckpoint(eager=True)

    max_cluster = (
        sdf
        .agg(F.max("CLUSTER"))
        .collect()[0][0]
    )

    offset = max_cluster + 1

    # Index end states without createDataFrame()
    end_state_window = Window.orderBy(
        F.col("end_state")
    )

    end_states_indexed = (
        end_states_df
        .withColumn(
            "end_state_indexed",
            (
                F.dense_rank()
                .over(end_state_window)
                - 1
                + F.lit(offset)
            ).cast("integer"),
        )
    )

    # 5. Final P matrix 
    P_sdf = (
        sdf
        .where(
            F.col("NEXT_CLUSTER").isNotNull()
        )
        .groupBy(
            "CLUSTER",
            "ACTION",
            "NEXT_CLUSTER",
        )
        .agg(
            F.count("*").alias("cnt")
        )
    )

    totals = (
        P_sdf
        .groupBy("CLUSTER", "ACTION")
        .agg(
            F.sum("cnt").alias("total")
        )
    )

    P_sdf = (
        P_sdf
        .join(
            totals,
            on=["CLUSTER", "ACTION"],
        )
        .withColumn(
            "prob",
            F.col("cnt") / F.col("total"),
        )
        .select(
            "CLUSTER",
            "ACTION",
            "NEXT_CLUSTER",
            "prob",
        )
    )

    # 6. Final R matrix 
    R_sdf = (
        sdf
        .groupBy("CLUSTER")
        .agg(
            F.mean("RISK").alias("RISK")
        )
        .select(
            "CLUSTER",
            "RISK",
        )
    )

    end_states_clusters = (
        end_states_indexed
        .select(
            F.col(
                "end_state_indexed"
            ).alias("CLUSTER"),
            F.col("Reward").alias("RISK"),
        )
        .distinct()
    )

    R_sdf = R_sdf.unionByName(
        end_states_clusters,
        allowMissingColumns=True,
    )

    # 7. Return pandas DataFrames 
    output_cols = (
        ["ID", "TIME"]
        + feature_cols
        + [
            "ACTION",
            "RISK",
            "CLUSTER",
            "NEXT_CLUSTER",
        ]
    )

    existing = [
        c
        for c in output_cols
        if c in sdf.columns
    ]

    df_trained_pd = (
        sdf
        .select(existing)
        .toPandas()
    )

    P_pd = P_sdf.toPandas()
    R_pd = R_sdf.toPandas()

    return df_trained_pd, P_pd, R_pd

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


def compute_most_likely_final_end_state(sdf, P_sdf, end_states_indexed):
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

    print(f"(Split) MDP most likely end state prediction accuracy: {accuracy:.4f}")
    print(f"(Split) MDP most likely end state confusion matrix:")
    for row in confusion_df.collect():
        print(
            f"(Split) Predicted: {row['predicted_final_cluster']}, True: {row['end_state_indexed']}, Count: {row['count']}"
        )

    return accuracy, confusion_df


def _recompute_next_cluster(sdf, end_states):
    """
    Compute NEXT_CLUSTER using window lead,
    filling each patient's final timestep with their indexed end state.
    """

    # Find where end-state cluster numbering should begin
    max_cluster = sdf.agg(
        F.max("CLUSTER")
    ).collect()[0][0]

    offset = max_cluster + 1

    # Assign each unique end_state an integer:
    #
    # offset, offset + 1, offset + 2, ...
    end_state_window = Window.orderBy("end_state")

    end_states_indexed = (
        end_states
        .withColumn(
            "end_state_indexed",
            (
                F.dense_rank().over(end_state_window)
                - 1
                + F.lit(offset)
            ).cast("integer")
        )
    )

    # Window over each patient's trajectory
    w = Window.partitionBy("ID").orderBy("TIME")

    # Remove old NEXT_CLUSTER if we're recomputing it
    if "NEXT_CLUSTER" in sdf.columns:
        sdf = sdf.drop("NEXT_CLUSTER")

    # Attach each patient's terminal state
    sdf_with_end_state = (
        sdf
        .join(
            end_states_indexed.select(
                "ID",
                F.col("end_state_indexed").alias("END_STATE")
            ),
            on="ID",
            how="left"
        )
    )

    # NEXT_CLUSTER =
    #   next timestep's CLUSTER, if one exists
    #   otherwise patient's terminal/end state
    return (
        sdf_with_end_state
        .withColumn(
            "NEXT_CLUSTER",
            F.coalesce(
                F.lead("CLUSTER").over(w),
                F.col("END_STATE")
            )
        )
        .drop("END_STATE")
    )


def _compute_incoherence(sdf, min_obs=500):
    print("(Split) deterministic count incoherence metric")
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
    print("(Split) JSD incoherence metric")
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


def split_worst_cluster_with_rf_and_kmeans(sdf, worst, prob_cols):
    """
    Splits the worst cluster/action using K-means and classifies remaining points with Random Forest.

    Args:
        sdf (DataFrame): Input Spark DataFrame.
        worst (dict): Dict with keys 'CLUSTER' and 'ACTION' indicating the worst cluster/action.
        prob_cols (list): List of column names for transition probabilities.

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
    else:
        # Only K-means results if no other actions in worst cluster
        all_worst_cluster_split = kmeans_labeled.withColumn(
            "final_sub_cluster", F.col("kmeans_label")
        )
    return all_worst_cluster_split


# ==========================
# SOLVING THE MDP
# ==========================

import pandas as pd
from .MDPTools import (
    makePandR_arrays,
    SolveMDP,
)

def solve_MDP(P_df, R_df):
    # solving the MDP
    P_df = P_df.dataframe().toPandas()
    print("P_df: ", P_df)
    R_df = R_df.dataframe().toPandas()
    R_df = R_df[["CLUSTER", "RISK"]].drop_duplicates()
    print("R_df: ", R_df)
    # df_trained = df_trained.dataframe().toPandas()

    p_series = P_df.set_index(["ACTION", "CLUSTER", "NEXT_CLUSTER"])[
        "prob"
    ].sort_index()
    r_series = R_df.set_index(["CLUSTER"])["RISK"].sort_index()
    #r_series.loc[12] = 100000000 # setting death reward to 100M
    print("p series: ", p_series)
    print("r series: ", r_series)
    P_arr, R_arr, counts_arr = makePandR_arrays(p_series, p_series, r_series, "min")
    print("P_arr: ", P_arr)
    print("R_arr: ", R_arr)
    V_E, pi_E, Q_E = SolveMDP(
        P_arr,
        R_arr,
        gamma=0.9,
        epsilon=10 ** (-10),
        p=False,
        prob="min",
        method="Value",
        threshold=float("inf"),
    )

    pi_E_df = pd.DataFrame(pi_E).reset_index().rename(columns={"index": "CLUSTER", "0": "ACTION"})
    Q_E_df = pd.DataFrame(Q_E)
    return pi_E_df, Q_E_df


def simulate_single_patient_trajectory(
    patient_id,
    starting_cluster,
    policy_dict,
    transition_dict,
    end_state_indices,
    max_steps=5000,
    random_seed=42
):
    """Simulate trajectory for a single patient - returns list of records."""
    if random_seed is not None:
        np.random.seed(random_seed + hash(patient_id) % 10000)
    
    trajectory = []
    current_cluster = int(starting_cluster)
    time_step = 0
    
    while time_step < max_steps and current_cluster not in end_state_indices:
        if current_cluster not in policy_dict:
            break
        action = int(policy_dict[current_cluster])
        print("cluster: ", current_cluster, "action: ", action)
        trajectory.append({
            "ID": patient_id,
            "TIME": time_step,
            "CLUSTER": current_cluster,
            "ACTION": action,
        })
        
        key = (current_cluster, action)
        if key not in transition_dict:
            break
            
        next_clusters, probs = transition_dict[key]
        current_cluster = int(np.random.choice(next_clusters, p=probs))
        time_step += 1
    
    # Add final state
    trajectory.append({
        "ID": patient_id,
        "TIME": time_step,
        "CLUSTER": current_cluster,
        "ACTION": None,
    })
    
    return trajectory

def simulate_trajectories(df_trained, P_df, pi_E_df, end_states):
    # Load as Spark DataFrames
    df_train_spark = df_trained.dataframe()
    P_spark = P_df.dataframe()
    pi_E_spark = pi_E_df.dataframe()
    end_state_df = end_states.dataframe()

    # Policy dictionary
    pi_E_pandas = (
        pi_E_spark
        .withColumnRenamed("0", "ACTION")
        .toPandas()
    )

    policy_dict_E = (
        pi_E_pandas
        .set_index("CLUSTER")["ACTION"]
        .to_dict()
    )

    # Transition dictionary
    P_pandas = P_spark.toPandas()

    transition_dict = {}

    for (cluster, action), group in P_pandas.groupby(["CLUSTER", "ACTION"]):
        next_clusters = group["NEXT_CLUSTER"].values
        probs = group["prob"].values

        probs = probs / probs.sum()

        transition_dict[(int(cluster), int(action))] = (
            next_clusters,
            probs
        )

    # End states
    max_real_cluster = (
        df_train_spark
        .agg(F.max("CLUSTER"))
        .collect()[0][0]
    )

    num_end_states = end_state_df.count()

    end_state_indices = list(
        range(
            max_real_cluster + 1,
            max_real_cluster + 1 + num_end_states
        )
    )

    # Starting states
    starting_states = (
        df_train_spark
        .filter(F.col("TIME") == 0)
        .select("ID", "CLUSTER")
    )

    # Collect only starting states to driver
    starting_states_pd = starting_states.toPandas()

    # Simulate trajectories
    trajectories = []

    for row in starting_states_pd.itertuples(index=False):
        trajectory = simulate_single_patient_trajectory(
            row.ID,
            row.CLUSTER,
            policy_dict_E,
            transition_dict,
            end_state_indices,
            max_steps=5000,
            random_seed=42
        )

        trajectories.extend(trajectory)

    # Return final pandas DataFrame
    result_E = pd.DataFrame(
        trajectories,
        columns=["ID", "TIME", "CLUSTER", "ACTION"]
    )

    return result_E