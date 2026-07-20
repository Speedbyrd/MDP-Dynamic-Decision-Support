from .model import MDP_model
import pandas as pd
import numpy as np


def train_mrl_on_sepsis(
    final_sepsis_df: pd.DataFrame,
    end_state_df: pd.DataFrame,
) -> "MDP_model":
    """
    Trains and returns a fitted MDP_model instance.
    final_sepsis_df columns (example): ['ID','TIME', ..., 'RISK','ACTION', ...]
    end_state_df: dataframe describing goal/end states
    """
    mdl = MDP_model()

    mdl.fit_stochastic_pred_prescrip(
        MDP_solver=solve_MDP_Robust,
        data=final_sepsis_df,
        end_state_df=end_state_df,
        pfeatures=15,  # features
        P_method=1,
        std_threshold=0.20,
        h=-1,
        gamma=1.0,
        max_k=20,
        distance_threshold=0.05,
        cv=5,
        th=0,
        eta=float("inf"),
        precision_thresh=1e-14,
        classification="RandomForestClassifier",
        split_classifier_params={"random_state": 0},
        clustering="KMeans",
        n_clusters=5,
        random_state=0,
        plot=True,
        optimize=True,
        verbose=False,
        df_init=None,
        min_obs=4444,
    )
    return mdl
