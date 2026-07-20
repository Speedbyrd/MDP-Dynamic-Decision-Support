# -*- coding: utf-8 -*-
"""
This file is intended to perform various testing measurements on the output of

the MDP Clustering Algorithm.

Created on Sun Apr 26 23:13:09 2020

@author: Amine
"""

#################################################################
# Load Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree

#################################################################


#################################################################
# Functions for Predictions


# predict_cluster() takes in a clustered dataframe df_new, the number of
# features pfeatures, and returns a prediction model m that predicts the most
# likely cluster from a datapoint's features
def predict_cluster(
    df_new, pfeatures  # dataframe: trained clusters
):  # int: # of features
    # the first 2 columns are ID and TIME so column 2 is the first feature
    # this would be safer if we took in a list of feature names and directly called the columns by their names
    X = df_new.iloc[:, 2 : 2 + pfeatures]
    y = df_new["CLUSTER"]

    params = {"max_depth": [4]}

    m = DecisionTreeClassifier()
    # m = RandomForestClassifier()

    m = GridSearchCV(estimator=m, param_grid=params, cv=5)
    # m = GridSearchCV(m, params, cv = 5, iid = True) # will return warning if 'iid' param not set to true

    # m = DecisionTreeClassifier(max_depth = 10)
    m.fit(X, y)
    return m


# predict_value_of_cluster() takes in MDP parameters, a cluster label, and
# and a list of actions, and returns the predicted value of the given cluster
# currently takes value of current cluster as well as next cluster
def predict_value_of_cluster(
    P_df, R_df, cluster, actions  # df: MDP parameters  # int: cluster number
):  # list: list of actions
    s = cluster
    v = R_df.loc[s]
    for a in actions:
        s = P_df.loc[s, a].values[0]
        v = v + R_df.loc[s]
    return v


# get_MDP() takes in a clustered dataframe df_new, and returns dataframes
# P_df and R_df that represent the parameters of the estimated MDP (if sink
# exists, it will be the last cluster and goes to itself)
def get_MDP(df_new):
    # removing None values when counting where clusters go
    death_state = df_new["NEXT_CLUSTER"].max()
    sink_state = death_state + 1
    readmit_state = death_state - 1
    discharge_state = death_state - 2
    df0 = df_new[df_new["NEXT_CLUSTER"] != "None"]
    # df0 = df_new
    transition_df = df0.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"])[
        "RISK"
    ].count()  # multi-index (3 index) series of counts. use transition_df[i,a,j]

    # next cluster given how most datapoints transition for the given action
    transition_df = transition_df.groupby(
        ["CLUSTER", "ACTION"]
    ).idxmax()  # returns the multi-index (cluster,action,next_cluster) with the highest count
    P_df = pd.DataFrame()
    P_df["NEXT_CLUSTER"] = transition_df.apply(lambda x: x[2])
    # print(P_df)

    R_df = df_new.groupby("CLUSTER")["RISK"].mean()  # a 1-index series object

    # # check if end state exists, if so make a sink node as the last cluster
    # if 'End' in list(P_df['NEXT_CLUSTER'].unique()):
    #     P_df = P_df.reset_index()

    #     # find goal cluster that leads to sink, then remove
    #     c = P_df.loc[P_df['NEXT_CLUSTER']=='End']['CLUSTER'].max() #the maximum cluster number that transitions to End?
    #     P_df = P_df.loc[P_df['NEXT_CLUSTER']!='End'] #removing the rows where next cluster is end

    #     # create rows in dataframe that say goal go to sink and sink go to sink
    #     # no matter which action you take, goal and sink node always transition to sink node
    #     s = P_df['CLUSTER'].max() + 1 #sink node?
    #     actions = P_df['ACTION'].unique() #list of actions observed in df
    #     df_end = []
    #     for a in actions:
    #         df_end.append([c, a, s])
    #         df_end.append([s, a, s])
    #     df_end = pd.DataFrame(df_end, columns = ['CLUSTER', 'ACTION', 'NEXT_CLUSTER'])

    #     P_df = P_df.append(df_end)
    #     P_df.sort_values(by=['CLUSTER','ACTION'], inplace=True)
    #     P_df.set_index(['CLUSTER','ACTION'], inplace=True)

    #     # set new reward node
    #     R_df = R_df.append(pd.Series([0], index=[s]))
    P_df = P_df.reset_index()
    actions = df_new["ACTION"].unique()
    df_end = []
    for a in actions:
        df_end.append([death_state, a, death_state])
        df_end.append([readmit_state, a, readmit_state])
        df_end.append([discharge_state, a, discharge_state])
        df_end.append([sink_state, a, sink_state])

    df_end = pd.DataFrame(df_end, columns=["CLUSTER", "ACTION", "NEXT_CLUSTER"])
    # print(df_end)
    P_df = P_df.append(df_end)
    # print(P_df)
    P_df.sort_values(by=["CLUSTER", "ACTION"], inplace=True)
    P_df.set_index(["CLUSTER", "ACTION"], inplace=True)
    # print(P_df)

    R_df.loc[discharge_state] = -30
    R_df.loc[readmit_state] = 25
    R_df.loc[death_state] = 30
    R_df.loc[sink_state] = 0
    R_df.sort_index(ascending=True, inplace=True)

    return P_df, R_df

def get_MDP_stochastic(df_new, end_state_rewards_df, method=1):
    # uses the columns for predicted probabilities of going to each next cluster
    # removing None values when counting where clusters go
    # df0 = df_new[df_new['NEXT_CLUSTER']!='None'] # ignoring none/end clusters for now
    # df_new = df_0
    # print("end states to rewards df get_MDP_stochastic: ", end_state_rewards_df)
    # print("end state rewards df columns get_MDP_stochastic: ", end_state_rewards_df.columns)
    # print("input df_new to get_MDP_stochastic: ", df_new)
    # print("columns of input df_new to get_MDP_stochastic: ", df_new.columns)
    last_real_state = df_new["CLUSTER"].max()
    # print("last real state index from get_MDP_stochastic: ", last_real_state)
    end_states = end_state_rewards_df["end_state"].unique()
    # print("end_state_rewards_df['end_state']", end_state_rewards_df["end_state"])
    # print("end state list from get_MDP_stochastic: ", end_states)
    end_state_inds = range(last_real_state + 1, last_real_state + len(end_states) + 1)
    # print("end state indices from get_MDP_stochastic: ", end_state_inds)
    end_state_rewards_df = end_state_rewards_df[["end_state", "Reward"]].set_index(
        "end_state"
    )
    # print("final end states to rewards df from get_MDP_stochastic: ", end_state_rewards_df)
    # death_state = df_new['NEXT_CLUSTER'].max()
    # sink_state = death_state + 1
    # readmit_state = death_state - 1
    # discharge_state = death_state - 2
    # ==========================================================================
    # empirical probabilities based on observed actual transitions (like the f(s,a)=argmax in the paper at bottom of page 30)
    # states = list(set(df_new['stratified_cluster']).union(set(df_new['next_stratified_cluster']))) # should be label-encoded now so the indices should be consecutive (not 1111,2222,4444)

    # short version of code
    transition_df = pd.DataFrame(
        df_new.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"])["RISK"].count()
    )  # multi-index (3 index) series of counts
    # transition_df[i,a,j] = number of points that took action a in state i and transitioned to state j
    cluster_action_df = pd.DataFrame(
        df_new.groupby(["CLUSTER", "ACTION"])["RISK"].count()
    )
    # cluster_action_df[i,a] = number of points who took action a in state i
    joined_df = transition_df.join(
        cluster_action_df, lsuffix="nextcount", rsuffix="totalcount"
    )
    P = joined_df["RISKnextcount"] / joined_df["RISKtotalcount"]

    actions = df_new["ACTION"].unique()
    # if method==2:
    clusters_list = list(set(df_new["CLUSTER"]).union(set(df_new["NEXT_CLUSTER"])))
    prob_vec_df = df_new.groupby(["CLUSTER", "ACTION"])[clusters_list].mean()
    P2 = pd.DataFrame(columns=["CLUSTER", "ACTION", "NEXT_CLUSTER", 0])
    P2.set_index(["CLUSTER", "ACTION", "NEXT_CLUSTER"], inplace=True)
    for idx in prob_vec_df.index:
        cluster = idx[0]
        action = idx[1]
        for col in prob_vec_df.columns:
            next_cluster = col
            prob = prob_vec_df.loc[idx][col]
            if prob > 0:
                P2.loc[cluster, action, next_cluster] = prob


    for a in actions:
        for end_state_idx in end_state_inds:
            P.loc[end_state_idx, a, end_state_idx] = 1
            P2.loc[end_state_idx, a, end_state_idx] = 1
            # P2.loc[readmit_state,a,readmit_state] = 1
            # P2.loc[discharge_state,a,discharge_state] = 1
            # P2.loc[sink_state,a,sink_state] = 1

    R_state = df_new.groupby("CLUSTER")["RISK"].mean()
    R_state = R_state.reset_index()  # makes both CLUSTER and RISK into columns
    R_state = R_state.set_index("CLUSTER")
    for end_state_idx in end_state_inds:
        R_state.loc[end_state_idx] = (
            end_state_rewards_df.loc[
                end_states[end_state_idx - last_real_state - 1], "Reward"
            ]
        ).astype(float)
    R_state.sort_index(ascending=True, inplace=True)
    print("Rewards df after get_MDP_stochastic: ", R_state)

    # but perhaps not all combinations of states and actions (and next states) are observed.. how to fill in missing ones?
    # R_state_action = df_new.groupby(['CLUSTER','ACTION'])['RISK'].mean() # rewards as a function of state and action
    # R_state_action_state  = df_new.groupby(['CLUSTER','ACTION','NEXT_CLUSTER'])['RISK'].mean() # rewards as a function of state and action and next_state

    # NOT P_df
    if method == 2:
        return P2, R_state
    else:
        return P, R_state  # , R_state_action, R_state_action_state


#################################################################
# Functions for Error


# training_value_error() takes in a clustered dataframe, and computes the
# E((\hat{v}-v)^2) expected error in estimating values (risk) given actions
# Returns a float of sqrt average value error per ID
def training_value_error(
    df_new,  # Output of algorithm
    gamma=1,  # discount factor
    relative=False,  # Output Raw error or RMSE ie ((\hat{v}-v)/v)^2
    h=5,
):  # Length of forecast. The error is computed on v_h = \sum_{t=h}^H v_t
    # if h = -1, we forecast the whole path
    # death_state = df_new['NEXT_CLUSTER'].max()
    # sink_state = death_state + 1
    # readmit_state = death_state - 1
    # discharge_state = death_state - 2

    df_new = df_new.sort_values(by=["ID", "TIME"], ascending=[True, True])
    E_v = 0
    P_df, R_df = get_MDP(df_new)
    df2 = df_new.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N_train = df2.shape[0]

    for i in range(N_train):
        index = df2["index"].iloc[i]
        # initializing first state for each ID
        cont = True

        if h == -1:
            t = 0

        else:
            H = -1
            # Computing Horizon H of ID i
            while cont:
                H += 1
                try:
                    df_new["ID"].loc[index + H + 1]
                except:
                    break
                if df_new["ID"].loc[index + H] != df_new["ID"].loc[index + H + 1]:
                    break
            t = H - h

        v_true = 0
        v_estim = 0
        s = df_new["CLUSTER"].loc[index + t]
        a = df_new["ACTION"].loc[index + t]

        # predicting path of each ID
        while cont:
            # if a == 'None':
            # break

            v_true = gamma * v_true + df_new["RISK"].loc[index + t]
            v_estim = gamma * v_estim + R_df.loc[s]
            # print('v_true', v_true, 'v_estim', v_estim)

            try:
                df_new["ID"].loc[index + t + 1]
            except:
                break
            if df_new["ID"].loc[index + t] != df_new["ID"].loc[index + t + 1]:
                break

            try:
                s = P_df.loc[s, a].values[0]
            # error raises in case we never saw a given transition in the data
            # except ValueError
            except:
                pass
                # print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)

            t += 1
            a = df_new["ACTION"].loc[index + t]

        if relative:
            E_v = E_v + ((v_true - v_estim) / v_true) ** 2
        else:
            E_v = E_v + (v_true - v_estim) ** 2
            # print('E_v diff:', v_true-v_estim)
    E_v = E_v / N_train
    return np.sqrt(E_v)


def stochastic_training_value_error(
    df_new,  # Output of algorithm
    end_state_rewards_df,
    P_method=1,
    gamma=1,  # discount factor
    relative=False,  # Output Raw error or RMSE ie ((\hat{v}-v)/v)^2
    h=-1,
    num_trajectories=1,
):  # Length of forecast. The error is computed on v_h = \sum_{t=h}^H v_t (H-h to H?my_model_s = model.MDP_model()
    # if h = -1, we forecast the whole path

    gamma = 1  # basically don't multiply by gamma
    # print("h from beginning: ", h)
    df_new = df_new.sort_values(by=["ID", "TIME"], ascending=[True, True])

    # print("clusters: ", df_new["CLUSTER"].unique())
    # print("next clusters: ", df_new["NEXT_CLUSTER"].unique())
    # sink_state = df_new["NEXT_CLUSTER"].max() + 1
    # print("sink state: ", sink_state)
    # death_state = df_new["NEXT_CLUSTER"].max()
    # readmitted_state = df_new["NEXT_CLUSTER"].max() - 1
    # discharged_state = df_new["NEXT_CLUSTER"].max() - 2

    last_real_state = df_new["CLUSTER"].max()
    end_states = end_state_rewards_df["end_state"].unique()
    end_state_inds = range(last_real_state + 1, last_real_state + len(end_states) + 1)
    # end_state_rewards_df = end_state_rewards_df[["end_state", "Reward"]].set_index(
    #     "end_state"
    # )

    v_true_total = 0
    P_df, R_df = get_MDP_stochastic(df_new, end_state_rewards_df, P_method)
    df2 = df_new.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N_train = df2.shape[0]

    for i in range(N_train):
        for k in range(num_trajectories):
            index = df2["index"].iloc[i]
            # print("index: ", index)
            # initializing first state for each ID
            cont = True

            if h == -1:
                t = 0

            else:
                H = -1
                # Computing Horizon H of ID i
                while cont:
                    H += 1
                    try:
                        df_new["ID"].loc[index + H + 1]
                    except:
                        break
                    if df_new["ID"].loc[index + H] != df_new["ID"].loc[index + H + 1]:
                        break
                t = H - h

            v_true = 0
            # v_estim = 0
            s = df_new["CLUSTER"].loc[index + t]
            # a = df_new['ACTION'].loc[index + t]

            # predicting path of each ID
            while cont:
                # if a == 'None':
                # break
                try:
                    df_new["ID"].loc[index + t + 1]
                except:
                    break
                if df_new["ID"].loc[index + t] != df_new["ID"].loc[index + t + 1]:
                    break

                v_true = gamma * v_true + df_new["RISK"].loc[index + t]
                # print("Reward of state s matches ", R_df[s] == R_df.loc[s])
                # v_estim = gamma*v_estim + R_df.loc[s]
                # print('v_true', v_true, 'v_estim', v_estim)

                try:
                    df_new["ID"].loc[index + t + 1]
                    s = df_new["CLUSTER"].loc[index + t + 1]
                except:
                    break

                if s in end_state_inds:
                    break
                t += 1
            v_true_total += v_true

    P_mat = pd.DataFrame(P_df)
    P_mat = P_mat.reset_index(drop=False)
    P_mat.rename(columns={0: "probability"}, inplace=True)
    
    # find the expected value of the data frame by having the points in each cluster-action transition to the predicted distribution
    # N_end_state_patients = 0 # number of patients who have reached their end state
    df0 = df_new.set_index("ID")
    expected_reward_total = 0
    time = 0
    current_df = df0[df0["TIME"] == time]  # make sure it's indexed by ID
    # print(current_df.head(10))
    # print(current_df.index)
    N_patients = current_df.shape[0]
    while N_patients > 0:
        time += 1
        next_df = df0[df0["TIME"] == time]  # next_df is not indexed by ID?
        # print(next_df.head(10))
        # print(next_df.index)
        next_patient_IDs = next_df.index  # is this a list? No but that's ok
        # print(next_patient_IDs)
        N_patients = next_df.shape[0]
        for cluster in current_df["CLUSTER"].unique():
            cluster_df = current_df[current_df["CLUSTER"] == cluster]
            # print("cluster: ", cluster)
            # print(cluster_df.shape)
            for action in cluster_df["ACTION"].unique():  # JUST DOING THE FIRST ONE
                cluster_action_df = cluster_df[cluster_df["ACTION"] == action]
                cluster_action_IDs = (
                    cluster_action_df.index
                )  # does this give you a list of icustay ids?
                # print("cluster: ", cluster, "action: ", action, "IDs: ", cluster_action_IDs)
                # print(next_df[['TIME','ACTION','CLUSTER','NEXT_CLUSTER']].loc[cluster_action_IDs].head(100))
                # print("action: ", action)
                # print(cluster_action_df.shape)
                N_cluster_action_patients = cluster_action_df.shape[0]
                next_state_probs = P_mat[
                    (P_mat["CLUSTER"] == cluster) & (P_mat["ACTION"] == action)
                ][["NEXT_CLUSTER", "probability"]]
                # print(next_state_probs)
                ind = 0
                # cluster_action_df = cluster_action_df.reset_index() # keeps old index (ID) column
                for next_state in next_state_probs["NEXT_CLUSTER"].unique():
                    # print("next state: ", next_state)
                    prob = next_state_probs[
                        next_state_probs["NEXT_CLUSTER"] == next_state
                    ][
                        "probability"
                    ].max()  # there's only 1 but max retrieves the number
                    # print("prob: ", prob)
                    N_next_state = round(N_cluster_action_patients * prob)
                    # calculate the reward accumulated here
                    # min makes sure the index doesn't run out of bounds
                    # NOT ALL PATIENTS ARE BEING ASSIGNED TO PREDICTED NEXT STATES WITH THIS ROUNDING SCHEME
                    next_state_IDs = cluster_action_IDs[
                        ind : min(ind + N_next_state, len(cluster_action_IDs))
                    ]
                    # doing intersection to ensure all next_state_IDs are indeed present in next_df
                    next_state_IDs = list(set(next_state_IDs) & set(next_patient_IDs))
                    # print("R df: ", R_df)
                    # print("next state: ", next_state)
                    # print("R_df.loc[next_state] : ", R_df.loc[next_state])
                    expected_reward_total = gamma * expected_reward_total + len(
                        next_state_IDs
                    ) * float(R_df.loc[next_state])
                    # print("next_state: ", next_state, "IDs: ", next_state_IDs)
                    ind += N_next_state
                    # print(N_next_state)
                    # next_df['CLUSTER'].loc[next_state_IDs] = next_state
                    next_df.loc[next_state_IDs, "CLUSTER"] = next_state
                    # print(next_df['CLUSTER'].loc[next_state_IDs])
                if ind < len(
                    set(cluster_action_IDs)
                ):  # ind is updated after the last assignment. So the index ind has not been assigned to a new cluster
                    try:
                        randomly_drawn_next_state = np.random.choice(
                            next_state_probs["NEXT_CLUSTER"],
                            1,
                            p=next_state_probs["probability"],
                        )[0]
                    except:
                        # print("trying to predict a transition never seen in the data")
                        randomly_drawn_next_state = (
                            cluster_action_df["NEXT_CLUSTER"].mode().max()
                        )  # .max is just to extract the number from the df
                    next_state_IDs = cluster_action_IDs[ind : len(cluster_action_IDs)]
                    # doing intersection to ensure all next_state_IDs are indeed present in next_df
                    next_state_IDs = list(set(next_state_IDs) & set(next_patient_IDs))
                    expected_reward_total = gamma * expected_reward_total + len(
                        next_state_IDs
                    ) * float(R_df.loc[randomly_drawn_next_state])
                    # next_df['CLUSTER'].loc[next_state_IDs] = randomly_drawn_next_state
                    next_df.loc[next_state_IDs, "CLUSTER"] = randomly_drawn_next_state
                    # print("randomly drawn next_state: ", randomly_drawn_next_state, "IDs: ", next_state_IDs)
        current_df = next_df

    print("total predicted reward: ", expected_reward_total)
    print("total true reward: ", v_true_total)

    return abs(expected_reward_total - v_true_total) / N_train  # mean absolute error?



# testing_value_error() takes in a dataframe of testing data, and dataframe of
# new clustered data, a model from predict_cluster function, and computes the
# expected value error given actions and a predicted initial cluster and time
# horizon h (if h = -1, we forecast the whole path)
# Returns a float of sqrt average value error per ID
def testing_value_error(
    df_test, df_new, model, pfeatures, gamma=1, relative=False, h=5  # discount factor
):
    E_v = 0
    P_df, R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N_test = df2.shape[0]

    df_test = df_test.assign(CLUSTER=model.predict(df_test.iloc[:, 2 : 2 + pfeatures]))

    for i in range(N_test):
        # initializing index of first state for each ID
        index = df2["index"].iloc[i]
        cont = True

        if h == -1:
            t = 0

        else:
            H = -1
            # Computing Horizon H of ID i
            while cont:
                H += 1
                try:
                    df_test["ID"].loc[index + H + 1]
                except:
                    break
                if df_test["ID"].loc[index + H] != df_test["ID"].loc[index + H + 1]:
                    break
            t = H - h

        v_true = 0
        v_estim = 0
        s = df_test["CLUSTER"].loc[index + t]
        a = df_test["ACTION"].loc[index + t]

        # predicting path of each ID
        while cont:
            v_true = gamma * v_true + df_test["RISK"].loc[index + t]
            v_estim = gamma * v_estim + R_df.loc[s]
            try:
                df_test["ID"].loc[index + t + 1]
            except:
                break
            if df_test["ID"].loc[index + t] != df_test["ID"].loc[index + t + 1]:
                break

            try:
                s = P_df.loc[s, a].values[0]
            # error raises in case we never saw a given transition in the data

            # except TypeError: # sometimes we see KeyError or IndexError...
            except:
                pass
                # print('WARNING: In training value evaluation, trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)

            t += 1
            a = df_test["ACTION"].loc[index + t]
        if relative:
            E_v = E_v + ((v_true - v_estim) / v_true) ** 2
        else:
            E_v = E_v + (v_true - v_estim) ** 2

    E_v = E_v / N_test
    return np.sqrt(E_v)


#################################################################


#################################################################
# Functions for R2 Values


# R2_value_training() takes in a clustered dataframe, and returns a float
# of the R-squared value between the expected value and true value of samples
# currently doesn't support horizon h specifications
def R2_value_training(df_new):
    # death_state = df_new['NEXT_CLUSTER'].max()
    # sink_state = death_state + 1
    # readmit_state = death_state - 1
    # discharge_state = death_state - 2
    E_v = 0
    P_df, R_df = get_MDP(df_new)
    # print(P_df)
    df2 = df_new.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N = df2.shape[0]
    V_true = []
    for i in range(N):
        # initializing starting cluster and values
        s = df2["CLUSTER"].iloc[i]
        a = df2["ACTION"].iloc[i]
        v_true = df2["RISK"].iloc[i]

        v_estim = R_df.loc[s]
        index = df2["index"].iloc[i]
        cont = True
        t = 1
        # iterating through path of ID
        while cont:
            v_true = v_true + df_new["RISK"].loc[index + t]
            try:
                df_new["ID"].loc[index + t + 1]
            except:
                break
            if df_new["ID"].loc[index + t] != df_new["ID"].loc[index + t + 1]:
                break

            try:
                s = P_df.loc[s, a].values[0]
            # error raises in case we never saw a given transition in the data
            # except TypeError:
            except:
                pass
                # print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_new["ACTION"].loc[index + t]
            v_estim = v_estim + R_df.loc[s]

            t += 1
        E_v = E_v + (v_true - v_estim) ** 2
        V_true.append(v_true)
    # defining R2 baseline & calculating the value
    E_v = E_v / N
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true - v_mean) ** 2) / N
    return max(1 - E_v / SS_tot, 0)


# R2_value_testing() takes a dataframe of testing data, a clustered dataframe,
# a model outputted by predict_cluster, and returns a float of the R-squared
# value between the expected value and true value of samples in the test set
# currently doesn't support horizon h specifications
def R2_value_testing(df_test, df_new, model, pfeatures):
    E_v = 0
    P_df, R_df = get_MDP(df_new)
    df2 = df_test.reset_index()
    df2 = df2.groupby(["ID"]).first()
    N = df2.shape[0]

    # predicting clusters based on features
    clusters = model.predict(df2.iloc[:, 2 : 2 + pfeatures])
    df2["CLUSTER"] = clusters

    V_true = []
    for i in range(N):
        s = df2["CLUSTER"].iloc[i]
        a = df2["ACTION"].iloc[i]
        v_true = df2["RISK"].iloc[i]

        v_estim = R_df.loc[s]
        index = df2["index"].iloc[i]
        cont = True
        t = 1
        while cont:
            v_true = v_true + df_test["RISK"].loc[index + t]

            try:
                s = P_df.loc[s, a].values[0]
            # error raises in case we never saw a given transition in the data
            # except TypeError:
            except:
                pass
                # print('WARNING: Trying to predict next state from state',s,'taking action',a,', but this transition is never seen in the data. Data point:',i,t)
            a = df_test["ACTION"].loc[index + t]

            v_estim = v_estim + R_df.loc[s]
            try:
                df_test["ID"].loc[index + t + 1]
            except:
                break
            if df_test["ID"].loc[index + t] != df_test["ID"].loc[index + t + 1]:
                break
            t += 1
        E_v = E_v + (v_true - v_estim) ** 2
        V_true.append(v_true)
    E_v = E_v / N
    V_true = np.array(V_true)
    v_mean = V_true.mean()
    SS_tot = sum((V_true - v_mean) ** 2) / N
    return max(1 - E_v / SS_tot, 0)


#################################################################


#################################################################
# Functions for Plotting and Visualization


# plot_features() takes in a dataframe and two features, and plots the data
# to illustrate the noise in each cluster
def plot_features(df, x, y, c="CLUSTER"):
    fig, ax = plt.subplots()
    df.plot(kind="scatter", x=x, y=y, c=c, cmap="tab20", ax=ax, s=5)

    #    import seaborn as sns
    #    sns.pairplot(x_vars=["FEATURE_1"], y_vars=["FEATURE_2"], data=df, hue="OG_CLUSTER", height=5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.axis("scaled")
    # plt.show()


# cluster_size() takes a dataframe, and returns the main statistics of each
# cluster in a dataframe
def cluster_size(df):
    df2 = df.groupby("CLUSTER")["RISK"].agg(["count", "mean", "std", "min", "max"])
    df2["rel"] = 100 * abs(df2["std"] / df2["mean"])
    df2["rel_mean"] = 100 * abs(df2["std"] / df["RISK"].mean())
    return df2


# next_clusters() takes a dataframe, and returns a chart showing the count and purity
# of the highest (most frequent) next_cluster for each cluster/action pair
# is designed for deterministic, NOT stochastic MRL
# Disregards those with 'NEXT_CLUSTER' = None, and returns a dataframe of the results
def next_clusters(df):
    df = df.loc[df["NEXT_CLUSTER"] != "None"]
    df2 = df.groupby(["CLUSTER", "ACTION", "NEXT_CLUSTER"])["RISK"].agg(["count"])
    df2["purity"] = df2["count"] / df.groupby(["CLUSTER", "ACTION"])["RISK"].count()
    df2.reset_index(inplace=True)
    idx = df2.groupby(["CLUSTER", "ACTION"])["count"].transform(max) == df2["count"]
    df_final = df2[idx].groupby(["CLUSTER", "ACTION"]).max()
    df_final["count"] = df2.groupby(["CLUSTER", "ACTION"])["count"].sum()
    return df_final


# def next_clusters_stratified(df):
#     df = df.loc[df['next_stratified_cluster']!='None']
#     df2 = df.groupby(['stratified_cluster', 'ACTION', 'next_stratified_cluster'])['RISK'].agg(['count'])
#     df2['purity'] = df2['count']/df.groupby(['stratified_cluster', 'ACTION'])['RISK'].count()
#     df2.reset_index(inplace=True)
#     idx = df2.groupby(['stratified_cluster', 'ACTION'])['count'].transform(max) == df2['count']
#     df_final = df2[idx].groupby(['stratified_cluster','ACTION']).max()
#     df_final['count'] = df2.groupby(['stratified_cluster', 'ACTION'])['count'].sum()
#     return df_final


# decision_tree_diagram() takes in a trained MDP model, outputs a pdf of
# the best decision tree, as well as other visualizations
def decision_tree_diagram(model):
    # assumes that m.m, the prediction model, is a GridSearchCV object
    dc = model.m.best_estimator_

    # creating the decision tree diagram in pdf:
    dot_data = tree.export_graphviz(
        dc, out_file=None, filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render("Decision_Tree_Diagram")

    return graph


# decision_tree_regions() takes a model and plots a visualization of the
# decision regions of two of the features (currently first and second)
def decision_tree_regions(model):
    dc = model.m.best_estimator_
    n_classes = model.df_trained["CLUSTER"].max()
    plot_step = 0.02

    plt.subplot()
    x_min = model.df_trained.iloc[:, 2].min() - 1
    x_max = model.df_trained.iloc[:, 2].max() + 1
    y_min = model.df_trained.iloc[:, 3].min() - 1
    y_max = model.df_trained.iloc[:, 3].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step)
    )

    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = dc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    for i in range(n_classes):
        idx = np.where(model.df_trained["CLUSTER"] == i)

        r = random.random()
        b = random.random()
        g = random.random()
        color = np.array([[r, g, b]])
        # colors = ['r', 'y', 'b']
        # color = colors[i%3]
        plt.scatter(
            model.df_trained.iloc[idx].iloc[:, 2],
            model.df_trained.iloc[idx].iloc[:, 3],
            c=color,
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

    plt.show()
    return


# model_trajectory() takes a trained model, the real transition function of
# the model f(x, u), the initial state x, and plots how the model's optimal
# policy looks like on the start state according to f1 and f2 two features
# indices e.g. x[f1] x[f2] plotted on the x and y axes, for n steps
def model_trajectory(
    m, f, x, f1=0, f2=None, n=50  # if f2 is none, only plot f1 over time
):
    states = []
    all_vecs = []
    if m.v is None:
        m.solve_MDP()

    if f2 != None:
        ys = [x[f2]]
        xs = [x[f1]]
    else:
        ys = [x[f1]]
        xs = range(n + 1)

    for i in range(n):
        # find current state and action
        s = m.m.predict(np.array(x).reshape(1, -1))
        # print(s)
        a = int(m.pi[s])
        # print(a)
        states.append([s, a])
        x_new = f(x, a)
        if x_new[0] == None:
            break

        if f2 != None:
            ys.append(x_new[f2])
            xs.append(x_new[f1])
        else:
            ys.append(x_new[f1])
        all_vecs.append(x_new)
        x = x_new
    print("states", states, flush=True)
    # TODO: not plot the sink
    xs = np.array(xs)
    ys = np.array(ys)

    u = np.diff(xs)
    v = np.diff(ys)
    pos_x = xs[:-1] + u / 2
    pos_y = ys[:-1] + v / 2
    norm = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
    # ax.set_xlabel('FEATURE_%i' %f1)
    # ax.set_ylabel('FEATURE_%i' %f2)

    # set plot limits if relevant
    # plt.ylim(-l+0.5, 0.5)
    # plt.xlim(-.5, l-0.5)
    plt.show()
    return xs, ys, all_vecs


# plot_CV_training() takes a model trained by cross validation, and plots the
# testing error, training, error, and incoherence on the same graph
def plot_CV_training(model):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("K meta-state space size")
    ax1.set_ylabel("Score")
    ax1.plot(model.CV_error_all["Training Error"], label="In-Sample")
    ax1.plot(model.CV_error_all["Testing Error"], label="Testing")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:red"
    ax2.set_ylabel("Number of Incoherences")  # we already handled the x-label with ax1
    ax2.plot(model.CV_error_all["Incoherence"], color=color, label="Incoherences")

    fig.legend()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


#################################################################


#################################################################
# Functions for Grid Testing (Predictions, Accuracy, Purity)


# get_predictions() takes in a clustered dataframe df_new, and maps each
# CLUSTER to an OG_CLUSTER that has the most elements
# Returns a dataframe of the mappings
def get_predictions(df_new):
    df0 = df_new.groupby(["CLUSTER", "OG_CLUSTER"])["ACTION"].count()
    df0 = df0.groupby("CLUSTER").idxmax()
    df2 = pd.DataFrame()
    df2["OG_CLUSTER"] = df0.apply(lambda x: x[1])
    return df2


# training_accuracy() takes in a clustered dataframe df_new, and returns the
# average training accuracy of all clusters (float) and a dataframe of
# training accuracies for each OG_CLUSTER
def training_accuracy(df_new):
    clusters = get_predictions(df_new)
    #    print('Clusters', clusters)

    # Tallies datapoints where the algorithm correctly classified a datapoint's
    # original cluster to be the OG_CLUSTER mapping of its current cluster
    accuracy = (
        clusters.loc[df_new["CLUSTER"]].reset_index()["OG_CLUSTER"]
        == df_new.reset_index()["OG_CLUSTER"]
    )
    # print(accuracy)
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame("Accuracy")
    accuracy_df["OG_CLUSTER"] = df_new.reset_index()["OG_CLUSTER"]
    accuracy_df = accuracy_df.groupby("OG_CLUSTER").mean()
    return (tr_accuracy, accuracy_df)


# testing_accuracy() takes in a testing dataframe df_test (unclustered),
# a df_new clustered dataset, a model from predict_cluster and
# Returns a float for the testing accuracy measuring how well the model places
# testing data into the right cluster (mapped from OG_CLUSTER), and
# also returns a dataframe that has testing accuracies for each OG_CLUSTER
def testing_accuracy(
    df_test,  # dataframe: testing data
    df_new,  # dataframe: clustered on training data
    model,  # function: output of predict_cluster, mdp.m
    pfeatures,
):  # int: # of features
    clusters = get_predictions(df_new)

    test_clusters = model.predict(df_test.iloc[:, 2 : 2 + pfeatures])
    df_test = df_test.assign(CLUSTER=test_clusters)

    accuracy = (
        clusters.loc[df_test["CLUSTER"]].reset_index()["OG_CLUSTER"]
        == df_test.reset_index()["OG_CLUSTER"]
    )
    # print(accuracy)
    tr_accuracy = accuracy.mean()
    accuracy_df = accuracy.to_frame("Accuracy")
    accuracy_df["OG_CLUSTER"] = df_test.reset_index()["OG_CLUSTER"]
    accuracy_df = accuracy_df.groupby("OG_CLUSTER").mean()
    return (tr_accuracy, accuracy_df)


# purity() takes a clustered dataframe and returns a dataframe with the purity
# of each cluster
def purity(df):
    su = pd.DataFrame(
        df.groupby(["CLUSTER"])["OG_CLUSTER"].value_counts(normalize=True)
    ).reset_index(level=0)
    su.columns = ["CLUSTER", "Purity"]
    return su.groupby("CLUSTER")["Purity"].max()


# generalization_accuracy() plots the training and testing accuracies as above
# for a given list of models and a test-set.
def generalization_accuracy(models, df_test, Ns):
    tr_accs = []
    test_accs = []
    for model in models:
        tr_acc, df = training_accuracy(model.df_trained)
        tr_accs.append(tr_acc)

        test_acc, df_t = testing_accuracy(
            df_test, model.df_trained, model.m, model.pfeatures
        )
        test_accs.append(test_acc)

    fig1, ax1 = plt.subplots()
    ax1.plot(Ns, tr_accs, label="Training Accuracy")
    ax1.plot(Ns, test_accs, label="Testing Accuracy")
    ax1.set_xlabel("N training data size")
    ax1.set_ylabel("Accuracy %")
    ax1.set_title("Model Generalization Accuracies")
    plt.legend()
    plt.show()
    return tr_accs, test_accs


# policy_accuracy() takes a trained model, and a df of optimal policies,
# and iterates through each row of the df to compare the model's predicted
# action with that of the real action taken in the data
# returns a percentage accuracy
def policy_accuracy(m, df):
    if m.v is None:
        m.solve_MDP()

    correct = 0
    df = df.loc[df["ACTION"] != "None"]
    # iterating through every line and comparing
    for index, row in df.iterrows():
        # predicted action:
        s = m.m.predict(np.array(row[2 : 2 + m.pfeatures]).reshape(1, -1))
        # s = m.df_trained.iloc[index]['CLUSTER']
        a = m.pi[s]

        # real action:
        a_true = row["ACTION"]
        if a == a_true:
            correct += 1
    total = df.shape[0]
    return correct / total


#################################################################
