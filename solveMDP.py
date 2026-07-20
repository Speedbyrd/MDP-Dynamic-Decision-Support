from transforms.api import transform, Input, Output, configure
from pyspark.sql import functions as F
import pandas as pd
from .MDPToolsRobust import (
    makePandR_arrays,
    SolveMDP,
)
import random
import numpy as np

@configure(profile=["DRIVER_MEMORY_EXTRA_EXTRA_LARGE", "EXECUTOR_MEMORY_MEDIUM"])
@transform(
    R_df=Input(
        "/path/to/R_df"
    ),
    P_df=Input(
        "/path/to/P_df"
    ),
    pi_E_df=Output(
        "/path/to/pi_E_df"
    ),
    Q_E_df=Output(
        "/path/to/Q_E_df"
    ),
)
def compute(
    P_df,
    R_df,
    pi_E_df,
    Q_E_df,
    ctx,
):
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

    pi_E = pd.DataFrame(pi_E).reset_index().rename(columns={"index": "CLUSTER", "0": "ACTION"})
    pi_E_df.write_dataframe(ctx.spark_session.createDataFrame(pi_E))
    Q_E_df.write_dataframe(ctx.spark_session.createDataFrame(pd.DataFrame(Q_E)))

    