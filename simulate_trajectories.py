from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import numpy as np

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


@configure(profile=["DRIVER_MEMORY_EXTRA_EXTRA_LARGE", "EXECUTOR_MEMORY_LARGE"])
@transform(
    simulated_trajectories_pi_E=Output(
        "/path/to/simulated_trajectories_pi_E"
    ),
    df_trained=Input("/path/to/df_trained"),
    end_states=Input("/path/to/end_states"),
    P_df=Input("/path/to/P_df"),
    pi_E_df=Input("/path/to/pi_E_df"),
)

def compute(df_trained, P_df, pi_E_df, end_states, simulated_trajectories_pi_E, ctx):
    spark = ctx.spark_session
    
    # Load data as Spark DataFrames
    df_train_spark = df_trained.dataframe()
    P_spark = P_df.dataframe()
    pi_E_spark = pi_E_df.dataframe()
    end_state_df = end_states.dataframe()
    
    # Prepare policy dictionary (small, can collect to driver)
    pi_E_pandas = pi_E_spark.toPandas()
    pi_E_pandas = pi_E_pandas.rename(columns={"0": "ACTION"})
    policy_dict_E = pi_E_pandas.set_index("CLUSTER")["ACTION"].to_dict()
    print("policy dict: ", policy_dict_E)
    
    # Prepare transition dictionary (small, can collect to driver)
    P_pandas = P_spark.toPandas()
    transition_dict = {}
    for (cluster, action), group in P_pandas.groupby(["CLUSTER", "ACTION"]):
        next_clusters = group["NEXT_CLUSTER"].values
        probs = group["prob"].values
        probs = probs / probs.sum()
        transition_dict[(int(cluster), int(action))] = (next_clusters, probs)
    
    # Determine end state indices
    max_real_cluster = df_train_spark.agg(F.max("CLUSTER")).collect()[0][0]
    num_end_states = end_state_df.count()
    end_state_indices = list(range(max_real_cluster + 1, max_real_cluster + 1 + num_end_states))
    # end_state_indices = list(range(8, 16))
    
    # Get starting states (TIME == 0) as Spark DataFrame
    starting_states = df_train_spark.where(F.col("TIME") == 0).select("ID", "CLUSTER")
    
    # Sample patients (10th partition)
    unique_ids = list(df_train_spark.select("ID").distinct().collect())
    #total_ids = len(unique_ids)
    unique_ids = [row["ID"] for row in unique_ids]
    
    starting_states = starting_states.where(F.col("ID").isin(unique_ids))
    
    # Broadcast small lookup dictionaries for efficiency
    policy_broadcast_E = spark.sparkContext.broadcast(policy_dict_E)
    print(policy_broadcast_E.value)
    transition_broadcast = spark.sparkContext.broadcast(transition_dict)
    end_states_broadcast = spark.sparkContext.broadcast(end_state_indices)
    
    # Define UDF for simulation
    from pyspark.sql.types import ArrayType
    
    def simulate_udf_E(patient_id, starting_cluster):
        return simulate_single_patient_trajectory(
            patient_id,
            starting_cluster,
            policy_broadcast_E.value,
            transition_broadcast.value,
            end_states_broadcast.value,
            max_steps=5000,
            random_seed=42
        )


    from pyspark.sql.functions import udf
    
    trajectory_schema = ArrayType(StructType([
        StructField("ID", StringType(), False),
        StructField("TIME", IntegerType(), False),
        StructField("CLUSTER", IntegerType(), False),
        StructField("ACTION", IntegerType(), True),
    ]))
    
    simulate_trajectory_udf_E = udf(simulate_udf_E, trajectory_schema)
    
    # Apply simulation in parallel across Spark executors
    simulated_E = starting_states.withColumn(
        "trajectory",
        simulate_trajectory_udf_E(F.col("ID"), F.col("CLUSTER"))
    )
    
    # Explode trajectories into individual rows
    result_E = simulated_E.select(F.explode("trajectory").alias("record")).select("record.*")
    
    # Write output
    simulated_trajectories_pi_E.write_dataframe(result_E)