# -*- coding: utf-8 -*-
"""
This file contains the functions to generate and perform the MDP clustering

algorithm on data for the MIT-Lahey Opioids project.

Created on Sun Mar  1 18:48:20 2020

@author: omars
new functions added by Angela
"""

#################################################################
# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from datetime import datetime
from tqdm import tqdm  # progress bar
import binascii
from copy import deepcopy
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Spark MLlib wrapper — used when spark_session is passed to splitter_stochastic
_SPARK_SESSION = None


def _make_classifier(classification, split_classifier_params, spark_session=None):
    """Create a classifier instance — Spark MLlib RF if spark_session is set, else sklearn."""
    if spark_session is not None and classification == "RandomForestClassifier":
        from .spark_rf_wrapper import SparkRFClassifier

        return SparkRFClassifier(
            spark_session,
            random_state=split_classifier_params.get("random_state", 0),
        )
    elif classification == "RandomForestClassifier":
        return RandomForestClassifier(**split_classifier_params)
    elif classification == "AdaBoostClassifier":
        return AdaBoostClassifier(**split_classifier_params)
    else:
        raise ValueError(f"Unknown classification: {classification}")


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold

# from xgboost import XGBClassifier
# from sklearn.model_selection import GridSearchCV
from scipy.stats import binom
from collections import Counter
from itertools import groupby
from operator import itemgetter

# from model import solve_MDP_Robust
from .testing import (
    R2_value_training,
    training_value_error,
    training_accuracy,
    predict_cluster,
    R2_value_testing,
    testing_value_error,
    testing_accuracy,
    next_clusters,
    stochastic_training_value_error,
    get_MDP_stochastic,
)

#################################################################


#################################################################
# Funtions for Initialization


# split_train_test_by_id() takes in a dataframe of all the data,
# returns Testing and Training dataset dataframes with the ratio of testing
# data defined by float test_ratio
def split_train_test_by_id(
    data,  # dataframe: all the data
    test_ratio,  # float: portion of data for testing
    id_column,
):  # str: name of identifying ID column
    def test_set_check(identifier, test_ratio):
        return binascii.crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2**32

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# initializeClusters() takes as input a dataframe,
# a clustering algorithm, a number of clusters n_clusters,
# and a random seed (optional) and returns a dataframe
# with two new columns 'CLUSTER' and 'NEXT_CLUSTER'
def initializeClusters(
    df,  # pandas dataFrame: MUST contain a "RISK" column
    end_state_df,  # pandas dataFrame: has 2 columns "ID" and "end_state"
    clustering="Agglomerative",  # string: clustering algorithm
    n_clusters=None,  # number of clusters
    distance_threshold=0.3,
    random_state=0,
):  # random seed for the clustering
    # end_state_df = end_state_df.replace('readmitted',2222)
    # end_state_df = end_state_df.replace('discharged',1111)
    # end_state_df = end_state_df.replace('dead',4444)
    end_state_df = end_state_df[["ID", "end_state"]].set_index("ID")
    # print("end state df initializeClusters: ", end_state_df)
    # print("end states initializeClusters: ", end_state_df["end_state"].unique())

    def replace_None_next_cluster(df_row):
        next_cluster = df_row["NEXT_CLUSTER"]
        id = df_row["ID"]
        if df_row["NEXT_CLUSTER"] == "None":
            next_cluster = end_state_df.loc[id, "end_state"]
        return next_cluster

    # sorts input dataframe by ID and time
    df = df.copy()
    df = df.sort_values(by=["ID", "TIME"])
    if clustering == "KMeans":
        print("KMeans")
        output = (
            KMeans(n_clusters=n_clusters, random_state=random_state)
            .fit(np.array(df.RISK).reshape(-1, 1))
            .labels_
        )
    elif clustering == "Agglomerative":
        print("Agglomerative")
        output = (
            AgglomerativeClustering(
                n_clusters=n_clusters, distance_threshold=distance_threshold
            )
            .fit(np.array(df.RISK).reshape(-1, 1))
            .labels_
        )
    elif clustering == "Birch":
        print("Birch")
        output = (
            Birch(n_clusters=n_clusters).fit(np.array(df.RISK).reshape(-1, 1)).labels_
        )
    else:
        output = LabelEncoder().fit_transform(np.array(df.RISK).reshape(-1, 1))
    df["CLUSTER"] = output
    df["CLUSTER"] = df["CLUSTER"].astype(int)
    df["NEXT_CLUSTER"] = df.groupby("ID")["CLUSTER"].shift(
        periods=-1, fill_value="None"
    )
    df["NEXT_CLUSTER"] = df.apply(lambda x: replace_None_next_cluster(x), axis=1)
    # print('cluster initialisation ran')
    # print(df.columns)
    # print(df)
    return df


#################################################################


#################################################################
# Function for the Iterations


# findContradiction() takes as input a dataframe and returns the tuple with
# initial cluster and action that have the most number of contradictions or
# (-1, -1) if no such cluster existss
# METHOD FOR FINDING CONTRADICTION:
# METHOD1: Let s,a be a state action pair. We count the number of occurences of
# different NEXT_CLUSTERS under action a starting from s. Let s' be NEXT_CLUSTER
# most elements from s went to under a. Let n2 be the number of elements that
# didn't go to s' under a. We chose (s,a) to maximize n2.
# METHOD2: Here n2 is chosen as the number of elements that went to the second
# most frequent NEXT_CLUSTER.
def findContradiction(
    df, th  # pandas dataFrame
):  # integer: threshold split size. cluster-action has to have this many points to be split
    X = df.loc[:, ["CLUSTER", "NEXT_CLUSTER", "ACTION"]]
    X = X[X.NEXT_CLUSTER != "None"]
    count = X.groupby(["CLUSTER", "ACTION"])["NEXT_CLUSTER"].nunique()
    # print(count)
    cluster_action_count = X.groupby(["CLUSTER", "ACTION"])["NEXT_CLUSTER"].count()
    # print(cluster_action_count)
    contradictions = set(count[list(count > 1)].index)  # what does this actually do
    large_enough_cluster_actions = set(
        cluster_action_count[list(cluster_action_count > th)].index
    )
    contradictions = list(contradictions.intersection(large_enough_cluster_actions))
    # print(len(contradictions))

    #    #METHOD 1
    #    if len(contradictions) > 0:
    #        ncontradictions = [sum(list(X.query('CLUSTER == @i[0]').query(
    #                'ACTION == @i[1]').groupby('NEXT_CLUSTER')['ACTION'].count().
    #            sort_values(ascending=False).ravel())[1:]) for i in contradictions]

    # #METHOD 2
    # if len(contradictions) > 0:
    #     ncontradictions = [sum(list(X.query('CLUSTER == @i[0]').query(
    #             'ACTION == @i[1]').groupby('NEXT_CLUSTER')['ACTION'].count().
    #         sort_values(ascending=False).ravel())[1:2]) for i in contradictions]
    #     #print(len(ncontradictions))
    #     if max(ncontradictions) > th:
    #         selectedCont = contradictions[ncontradictions.index(
    #                 max(ncontradictions))]
    #         return(selectedCont)
    # METHOD 3
    if len(contradictions) > 0:
        ncontradictions = [
            sum(
                list(
                    X.query("CLUSTER == @i[0]")
                    .query("ACTION == @i[1]")
                    .groupby("NEXT_CLUSTER")["ACTION"]
                    .count()
                    .sort_values(ascending=False)
                    .ravel()
                )[1:2]
            )
            for i in contradictions
        ]
        # print(ncontradictions)
        # looks at the proportion of points rather than raw # of points
        pcontradictions = [
            ncontradictions[ind] / cluster_action_count[contradictions[ind]]
            for ind in range(len(ncontradictions))
        ]
        # print(pcontradictions)
        selectedCont = contradictions[pcontradictions.index(max(pcontradictions))]
        return selectedCont

    return (-1, -1)


# contradiction() outputs one found contradiction given a dataframe,
# a cluster and a an action or (None, None) if none is found
# returns the action and (list of ) most frequently observed next cluster
def contradiction(
    df, i, a  # pandas dataFrame  # integer: initial clusters
):  # integer: action taken
    nc = list(
        df.query("CLUSTER == @i")
        .query("ACTION == @a")
        .query('NEXT_CLUSTER != "None"')["NEXT_CLUSTER"]
    )
    if len(nc) == 1:
        return (None, None)
    else:
        return a, multimode(nc)[0]


# multimode() returns a list of the most frequently occurring values.
# Will return more than one result if there are multiple modes
# or an empty list if *data* is empty.
def multimode(data):
    counts = Counter(iter(data)).most_common()
    maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
    return list(map(itemgetter(0), mode_items))


# split() takes as input a dataframe, an initial cluster, an action, a target
# cluster that is a contradiction c, then number of features,
# and an iterator k (that is the indexer of the next cluster), as well as the
# predictive classification algorithm used
# Returns a new dataframe with the contradiction resolved, and the best fit score
# for the splitting model (if GridSearch used)
def split(
    df,  # pandas dataFrame
    i,  # integer: initial cluster (to be split)
    a,  # integer: action taken
    c,  # integer: target cluster (cluster most points went to)
    pfeatures,  # integer: number of features
    k,  # integer: indexer for new next cluster
    classification="LogisticRegression",  # string: classification aglo
    split_classifier_params={"random_state": 0},
):  # dict: of classifier params
    g1 = df[
        (df["CLUSTER"] == i) & (df["ACTION"] == a) & (df["NEXT_CLUSTER"] == c)
    ]  # points that went to (target) next cluster
    g2 = df[
        (df["CLUSTER"] == i)
        & (df["ACTION"] == a)
        & (df["NEXT_CLUSTER"] != c)
        & (  # points that did not go to (target) next cluster
            df["NEXT_CLUSTER"] != "None"
        )
    ]
    g3 = df[
        (df["CLUSTER"] == i)
        & (
            ((df["ACTION"] == a) & (df["NEXT_CLUSTER"] == "None"))
            | (df["ACTION"] != a)  # points that did not take action a
        )
    ]
    groups = [g1, g2, g3]
    data = {}

    for j in range(len(groups)):
        d = pd.DataFrame(groups[j].iloc[:, 2 : 2 + pfeatures].values.tolist())

        data[j] = d

    data[0].insert(
        data[0].shape[1], "GROUP", np.zeros(data[0].shape[0])
    )  # inserting a column that labels all 0s
    data[1].insert(
        data[1].shape[1], "GROUP", np.ones(data[1].shape[0])
    )  # inserting a column that labels all 1s

    training = pd.concat([data[0], data[1]])

    tr_X = training.iloc[:, :-1]
    tr_y = training.iloc[:, -1:]

    if classification == "LogisticRegression":
        m = LogisticRegression(**split_classifier_params)
    elif classification == "LogisticRegressionCV":
        m = LogisticRegressionCV(**split_classifier_params)
    elif classification == "DecisionTreeClassifier":
        m = DecisionTreeClassifier(**split_classifier_params)
        # params = {
        # 'max_depth': [3, None]
        # }
        # m = GridSearchCV(m, params,cv = 5)
    elif classification == "RandomForestClassifier":
        m = RandomForestClassifier(**split_classifier_params)
        # params = {
        # 'max_depth': [3, None]
        # }
        # m = GridSearchCV(m, params,cv = 5)
    # elif classification == 'XGBClassifier':
    # m = XGBClassifier()
    elif classification == "MLPClassifier":
        m = MLPClassifier(**split_classifier_params)
    elif classification == "AdaBoostClassifier":
        m = AdaBoostClassifier(**split_classifier_params)
    else:
        m = LogisticRegression(**split_classifier_params)

    m.fit(tr_X, tr_y.values.ravel())
    try:
        score = m.best_score_
    except:
        score = None

    ids = g2.index.values

    test_X = data[2]  # group 3

    if len(test_X) != 0:
        Y = m.predict(test_X)
        g3.insert(g3.shape[1], "GROUP", Y.ravel())
        id2 = g3.loc[g3["GROUP"] == 1].index.values
        ids = np.concatenate(
            (ids, id2)
        )  # indices that did not go to target next cluster

    df.loc[df.index.isin(ids), "CLUSTER"] = k  # indexer for new cluster
    # newids = ids-1
    # df.loc[(df.index.isin(newids)) &
    #        (df['ID']== df['ID'].shift(-1)), 'NEXT_CLUSTER'] = k

    return df, score


# splitter() is the wrap-up function. Takes the below parameters and
# performs the algorithm until all contradictions are
# resolved or until the max number of iterations is reached
# Plots the trajectory of testing metrics during splitting process
# Returns the final resulting dataframe, as well as incoherences, errors,
# and the dataframe with the optimal split
def splitter(
    df,  # pandas dataFrame
    end_state_df,
    pfeatures,  # integer: number of features
    th,  # integer: threshold for minimum split
    eta=25,  # incoherence threshold for splits
    precision_thresh=1e-14,  # precision threshold when considering new min value error
    df_test=None,  # df_test provided for cross validation
    testing=False,  # True if we are cross validating
    max_k=6,  # int: max number of clusters
    classification="LogisticRegression",  # string: classification alg
    split_classifier_params={"random_state": 0},  # dict: classification params
    h=5,
    gamma=1,
    verbose=False,
    n=-1,
    plot=False,
):
    # initializing lists for error & accuracy data
    end_state_df = end_state_df.replace("readmitted", 2222)  # make this more general
    end_state_df = end_state_df.replace("discharged", 1111)
    end_state_df = end_state_df.replace("dead", 4444)
    end_state_df = end_state_df[["ID", "end_state"]].set_index(
        "ID"
    )  # column named ID gets deleted

    def replace_None_next_cluster(df_row):
        next_cluster = df_row["NEXT_CLUSTER"]
        ID = df_row["ID"]
        if df_row["NEXT_CLUSTER"] == "None":
            next_cluster = end_state_df.loc[ID, "end_state"]
        return next_cluster

    df.reset_index(drop=True, inplace=True)

    training_R2 = []
    testing_R2 = []
    training_acc = []
    testing_acc = []
    testing_error = []
    training_error = []

    incoherences = []
    split_scores = []
    thresholds = []

    # determine if the problem has OG cluster
    if "OG_CLUSTER" in df.columns:
        grid = True
    else:
        grid = False

    features = list(
        set(df.columns).difference(
            set(["ID", "TIME", "RISK", "ACTION", "CLUSTER", "NEXT_CLUSTER"])
        )
    )
    print("correct number of features", pfeatures == len(features))

    k = int(df["CLUSTER"].nunique())  # initial number of clusters
    nc = k  # number of clusters

    df_new = deepcopy(df)

    # storing optimal df
    best_df = None
    opt_k = None
    min_error = float("inf")

    # backup values in case threshold fails
    backup_min_error = float("inf")
    backup_df = None
    backup_opt_k = None

    # Setting progress bar--------------
    split_bar = tqdm(range(int(max_k - k)))
    split_bar.set_description("Splitting...")
    # Setting progress bar--------------
    for i in split_bar:
        split_bar.set_description("Splitting... |#Clusters:%s" % (nc))
        cont = False
        c, a = findContradiction(df_new, th)
        if c != -1:
            # finding contradictions and splitting
            a, b = contradiction(df_new, c, a)  # b is the cluster most actions went to

            if verbose:
                print(
                    "Cluster splitted",
                    c,
                    "| Action causing contradiction:",
                    a,
                    "| Cluster most elements went to:",
                    b,
                )
            df_new, score = split(
                df_new, c, a, b, pfeatures, nc, classification, split_classifier_params
            )
            split_scores.append(score)

            df_new = df_new.sort_values(by=["ID", "TIME"], ascending=[True, True])
            df_new["NEXT_CLUSTER"] = df_new.groupby("ID")["CLUSTER"].shift(
                periods=-1, fill_value="None"
            )
            df_new["NEXT_CLUSTER"] = df_new.apply(
                lambda x: replace_None_next_cluster(x), axis=1
            )  # replaces None with 1111,2222,4444

            le = preprocessing.LabelEncoder()
            clusters = list(set(df_new["CLUSTER"]).union(set(df_new["NEXT_CLUSTER"])))
            le.fit(clusters)
            df_new["CLUSTER"] = le.transform(df_new["CLUSTER"])
            # df_new["NEXT_CLUSTER"] = df_new["NEXT_CLUSTER"].astype(int)
            df_new["NEXT_CLUSTER"] = le.transform(df_new["NEXT_CLUSTER"])
            print(df_new["CLUSTER"].value_counts())

            # calculate incoherences - stochastic MRL has its own way of doing this
            next_clus = next_clusters(df_new)
            # calculates number of incoherences for each cluster-action pair
            # next_clus['incoherence'] = (1-next_clus['purity'])*next_clus['count']
            next_clus["incoherence"] = 1 - next_clus["purity"]
            next_clus.reset_index(inplace=True)
            # sum the incoherences in each cluster
            next_clus = next_clus.groupby("CLUSTER").sum()
            # find the cluster with the highest number of incoherences
            max_inc = next_clus["incoherence"].max()
            incoherences.append(max_inc)

            # error and accuracy calculations
            R2_train = R2_value_training(df_new)
            training_R2.append(R2_train)
            train_error = training_value_error(df_new, gamma, relative=False, h=h)
            training_error.append(train_error)

            if grid:
                train_acc = training_accuracy(df_new)[0]
                training_acc.append(train_acc)

            if testing:
                model = predict_cluster(df_new, pfeatures)
                R2_test = R2_value_testing(df_test, df_new, model, pfeatures)
                testing_R2.append(R2_test)
                test_error = testing_value_error(
                    df_test, df_new, model, pfeatures, gamma, relative=False, h=h
                )
                testing_error.append(test_error)

                if grid:
                    test_acc = testing_accuracy(df_test, df_new, model, pfeatures)[0]
                    testing_acc.append(test_acc)

            # printing error and accuracy values
            if verbose:
                print("training value R2:", R2_train)
                print("training value error:", train_error)
                if grid:
                    print("training accuracy:", train_acc)
                if testing:
                    print("testing value R2:", R2_test)
                    print("testing value error:", test_error)
                    if grid:
                        print("testing accuracy:", test_acc)
            # print('predictions:', get_predictions(df_new))
            # print(df_new.head())

            # update optimal dataframe if inc threshold and min error met
            # threshold calculated using eta * sqrt(number of datapoints) /
            # number of clusters
            threshold = eta * df_new.shape[0] ** 0.5 / (nc + 1)
            thresholds.append(threshold)
            if verbose:
                print("threshold:", threshold, "max_incoherence:", max_inc)

            # only update the best dataframe if training error is smaller
            # than previous training error by at least precision_thresh,
            # and also if maximum incoherence is lower than calculated threshold
            if max_inc < threshold and train_error < (min_error - precision_thresh):
                min_error = train_error
                best_df = df_new.copy()
                opt_k = nc + 1
                if verbose:
                    print("new opt_k", opt_k)

            # code for storing optimal clustering even if incorrect incoherence
            # threshold is chosen and nothing passes threshold; to prevent
            # training interruption
            elif opt_k == None and train_error < (backup_min_error - precision_thresh):
                backup_min_error = train_error
                backup_df = df_new.copy()
                backup_opt_k = nc + 1

            cont = True
            nc += 1
        if not cont:
            break
        if nc >= max_k:
            if verbose:
                print("Optimal # of clusters reached")
            break

        # plot every 20 iterations

        # if plot:
        #     if i%20 == 0:
        #         its = np.arange(k+1, nc+1)
        #         fig2, ax2 = plt.subplots()
        #         ax2.plot(its, training_error, label = "Training Error")
        #         if testing:
        #             ax2.plot(its, testing_error, label = "Testing Error")
        #         if n>0:
        #             ax2.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
        #         ax2.set_ylim(0)
        #         ax2.set_xlabel('# of Clusters')
        #         ax2.set_ylabel('Value error')
        #         ax2.set_title('Value error by number of clusters')
        #         ax2.legend()
        #         plt.show()

    # in the case that threshold prevents any values from passing, use backup
    if opt_k == None:
        opt_k = backup_opt_k
        best_df = backup_df
        min_error = backup_min_error

    # plotting functions
    ## Plotting accuracy and value R2
    its = np.arange(k + 1, nc + 1)
    if plot:
        if grid:
            fig1, ax1 = plt.subplots()
            # ax1.plot(its, training_R2, label= "Training R2")
            ax1.plot(its, training_acc, label="Training Accuracy")
            if testing:
                ax1.plot(its, testing_acc, label="Testing Accuracy")
                # ax1.plot(its, testing_R2, label = "Testing R2")
            if n > 0:
                ax1.axvline(
                    x=n, linestyle="--", color="r"
                )  # Plotting vertical line at #cluster =n
            ax1.set_ylim(0, 1)
            ax1.set_xlabel("# of Clusters")
            ax1.set_ylabel("R2 or Accuracy %")
            ax1.set_title("R2 and Accuracy During Splitting")
            ax1.legend()
        ## Plotting value error E((v_est - v_true)^2)
        fig2, ax2 = plt.subplots()
        norm_max = max(incoherences)
        ax2.plot(its, training_error, label="Training Error")
        ax2.plot(its, np.array(incoherences) / norm_max, label="Max Incoherence")
        ax2.plot(its, np.array(thresholds) / norm_max, "r-", label="Threshold")
        if testing:
            ax2.plot(its, testing_error, label="Testing Error")
        if n > 0:
            ax2.axvline(
                x=n, linestyle="--", color="r"
            )  # Plotting vertical line at #cluster =n
        ax2.set_ylim(0)
        ax2.set_xlabel("# of Clusters")
        ax2.set_ylabel("Value error")
        ax2.set_title("Value error by number of clusters")
        ax2.legend()
        plt.show()

    df_train_error = pd.DataFrame(
        list(zip(its, training_error)), columns=["Clusters", "Error"]
    )
    df_incoherences = pd.DataFrame(
        list(zip(its, incoherences)), columns=["Clusters", "Incoherences"]
    )
    if testing:
        df_test_error = pd.DataFrame(
            list(zip(its, testing_error)), columns=["Clusters", "Error"]
        )
        return (
            df_new,
            df_incoherences,
            df_train_error,
            df_test_error,
            best_df,
            opt_k,
            split_scores,
        )
    return (
        df_new,
        df_incoherences,
        df_train_error,
        testing_error,
        best_df,
        opt_k,
        split_scores,
    )

# splitter() is the wrap-up function. Takes the below parameters and
# performs the algorithm until all contradictions are
# resolved or until the max number of iterations is reached
# Plots the trajectory of testing metrics during splitting process
# Returns the final resulting dataframe, as well as incoherences, errors,
# and the dataframe with the optimal split
# inputted dataframe already has clusters and next clusters
# df MUST USE DEFAULT INDEXING
def splitter_stochastic(
    train_df,  # pandas dataFrame #with 'ClUSTER' and 'NEXT_CLUSTER columns TRAINING DF
    val_df,
    end_state_df,
    # end_state_list,
    p,  # integer: number of features
    threshold=0,  # threshold for minimum standard deviation to split
    P_method=1,
    eta=25,  # not used
    precision_thresh=1e-14,  # precision threshold when considering new min value error
    df_test=None,  # df_test provided for cross validation
    testing=False,  # True if we are cross validating
    max_iter=56,  # int: max number of cluster splits
    classification="RandomForestClassifier",  # string: classification alg
    split_classifier_params={
        "random_state": 0
    },  # dict: classification params #not used
    h=-1,  # idk what this is, not used
    gamma=1,  # idk what this is, not used
    verbose=False,  # idk what this is, not used
    n_param=-1,  # idk what this is, not used
    plot=False,
    min_obs=2000,
    incoherence_metric="std",
    spark_session=None,  # When provided, uses Spark MLlib RF for distributed training
):
    print("incoherence metric: ", incoherence_metric)
    print("min # points for cluster to be split: ", min_obs)
    df = train_df.copy()

    # print("initial clusters: ", df["CLUSTER"].unique())
    # print("initial next clusters: ", df["NEXT_CLUSTER"].unique())

    # we're gonna use the same method as the deterministic algorithm in deciding which cluster to split
    # end_state_df = end_state_df.replace('readmitted',2222) #make this more general
    # end_state_df = end_state_df.replace('discharged',1111)
    # end_state_df = end_state_df.replace('dead',4444)

    end_state_df = end_state_df[["ID", "end_state", "Reward"]].set_index(
        "ID"
    )  # column named ID gets deleted
    end_state_rewards_df = end_state_df.groupby(["end_state"])["Reward"].mean()
    end_state_rewards_df = (
        end_state_rewards_df.reset_index()
    )  # makes both end_state and Reward into columns again

    def replace_None_next_cluster(df_row):
        next_cluster = df_row["NEXT_CLUSTER"]
        ID = df_row["ID"]
        if df_row["NEXT_CLUSTER"] == "None":
            next_cluster = end_state_df.loc[ID, "end_state"]
        return next_cluster

    def JSD(Q, P):
        if len(Q) == len(P):
            M = (Q + P) / 2
            Qsum = 0
            for i in range(len(Q)):
                if Q[i] != 0:
                    Qsum += Q[i] * math.log(Q[i] / M[i])
            Psum = 0
            for i in range(len(P)):
                if P[i] != 0:
                    Psum += P[i] * math.log(P[i] / M[i])

            return (Psum + Qsum) / 2
        else:
            print("unequal lengths")

    df.reset_index(drop=True, inplace=True)

    # the output dataframe has updated CLUSTER and NEXT_CLUSTER columns as well as predicted probabilities of ending up in each of the clusters next

    # initializing lists for error & accuracy data
    training_R2_1_step = []
    testing_R2_1_step = []
    #     testing_R2 = []
    #     training_acc = []
    #     testing_acc = []
    testing_error_1_step = []
    training_error_whole = []
    training_error_1_step = []
    #
    incoherences = []
    #     split_scores = []

    init_k = int(df["CLUSTER"].nunique())  # initial number of clusters
    nc = init_k  # number of clusters
    print("initial number of clusters: ", nc)

    # df_new = deepcopy(df)

    # storing optimal df
    best_df = None
    opt_k = None
    min_error = float("inf")

    # backup values in case threshold fails

    #     backup_min_error = float('inf')
    #     backup_df = None
    #     backup_opt_k = None

    features = list(
        set(df.columns).difference(
            set(["ID", "TIME", "RISK", "ACTION", "CLUSTER", "NEXT_CLUSTER"])
        )
    )
    # print("full set of columns: ", df.columns)
    print("correct number of features", p == len(features))
    print("number of points per cluster: ", df["CLUSTER"].value_counts())

    # assumes number of columns besides features is 6 (ID, TIME, ACTION, RISK, CLUSTER, NEXT_CLUSTER)
    incoherence_level = 1000000000000  # initiate with a high incoherence level
    counter = max(df["CLUSTER"])  # highest cluster number
    n_iter = 0
    while (
        incoherence_level >= threshold and n_iter <= max_iter
    ):  # while the standard deviation is still high and we haven't split the max number of times yet
        n_iter += 1
        print("iteration number: ", n_iter)

        le = preprocessing.LabelEncoder()
        clusters = list(set(df["CLUSTER"]).union(set(df["NEXT_CLUSTER"])))
        print("clusters: ", clusters)
        le.fit(clusters)
        df["CLUSTER"] = le.transform(df["CLUSTER"])
        print("unique clusters: ", df["CLUSTER"].unique())
        # df["NEXT_CLUSTER"] = df["NEXT_CLUSTER"].astype(int)
        df["NEXT_CLUSTER"] = le.transform(df["NEXT_CLUSTER"])
        print("unique next clusters: ", df["NEXT_CLUSTER"].unique())

        model = _make_classifier(classification, split_classifier_params, spark_session)
        ## fits classifier to predict (probabilistically) the next cluster based on features and action
        X = df.loc[:, features + ["ACTION"]]
        y = df["NEXT_CLUSTER"].astype("int")
        model.fit(X, y)
        ## appends predicted probability columns for being in each cluster onto the df
        df = df.iloc[:, : (6 + p)].join(
            pd.DataFrame(model.predict_proba(df.loc[:, features + ["ACTION"]]))
        )  # why :6+p? Some problem with columns overlapping idk
        # print("df with predicted probabilites: ",df.isnull().any())

        train_error_1_step, R2_1_step = training_reward_error(
            df, end_state_rewards_df, P_method
        )
        train_error_whole = stochastic_training_value_error(
            df, end_state_rewards_df, P_method
        )
        test_error_1_step, test_R2_1_step = testing_reward_error(
            df, val_df, p, end_state_df, end_state_rewards_df, P_method
        )
        print(
            "num clusters: ",
            nc,
            "1 step value prediction R2: ",
            R2_1_step,
            "1 step validation R2: ",
            test_R2_1_step,
        )
        testing_error_1_step.append(test_error_1_step)
        training_error_whole.append(train_error_whole)
        training_error_1_step.append(train_error_1_step)
        training_R2_1_step.append(R2_1_step)
        testing_R2_1_step.append(test_R2_1_step)
        ## for each cluster-action group, compute the sum of std (across whole group) in probability of going to each next cluster
        grouped = (
            df.groupby(["CLUSTER", "ACTION"])[df.columns[(6 + p) :]]
            .std()
            .sum(axis=1)
            .reset_index()
        )  # every column from 6+p to end is a predicted probability of going to a cluster
        grouped.columns = list(grouped.columns[:-1]) + [
            "Sum_std"
        ]  # renaming last column "Sum_std"

        # grouped = pd.DataFrame(grouped)
        grouped["DET_INCOHERENCE"] = 0
        grouped["INFO_RADIUS"] = 0
        grouped["Count"] = (
            df.groupby(["CLUSTER", "ACTION"])["TIME"].count().reset_index()["TIME"]
        )
        for (
            ind
        ) in (
            grouped.index
        ):  # for each cluster, action pair, calculating the incoherence/dissimilarity in probability transitions
            cluster, action = grouped.loc[ind, ["CLUSTER", "ACTION"]]
            X = df.query("CLUSTER == @cluster and ACTION == @action")[
                df.columns[(6 + p) :]
            ]  # only the probability columns
            k = X.shape[0]
            info_radius_sum = 0
            avg_prob_vec = X.mean().values
            for i in range(k):
                info_radius_sum += JSD(X.values[i, :], avg_prob_vec)
            info_radius = info_radius_sum / k
            # print(Xmat.shape, "count: ", grouped.loc[ind, 'Count'])
            Xmat = X.to_numpy()
            det_incoherence = math.sqrt(
                abs(np.linalg.det(np.matmul(Xmat.transpose(), Xmat)))
            )
            # print(det_incoherence)
            grouped.loc[ind, "DET_INCOHERENCE"] = det_incoherence
            grouped.loc[ind, "INFO_RADIUS"] = info_radius

        try:
            ## finds the cluster-action group (that has enough observations) with the highest summed std (most incoherent) in order to split it into 2 clusters
            # print(grouped.query('Count >= @min_obs').sort_values('Sum_std', ascending=False))
            max_std_cluster, max_std_action, max_std, _, _, max_std_count = (
                grouped.query("Count >= @min_obs")
                .sort_values("Sum_std", ascending=False)
                .iloc[0]
            )
            print(
                "max summed std cluster: ",
                max_std_cluster,
                ", action: ",
                max_std_action,
                ", summed std: ",
                max_std,
                ", count: ",
                max_std_count,
            )
        except ValueError:
            print("break")
            break

        try:
            ## finds the cluster-action group (that has enough observations) with the highest determinant (most incoherent) in order to split it into 2 clusters
            # print(grouped.query('Count >= @min_obs').sort_values('DET_INCOHERENCE', ascending=False))
            max_det_cluster, max_det_action, _, max_det, _, max_det_count = (
                grouped.query("Count >= @min_obs")
                .sort_values("DET_INCOHERENCE", ascending=False)
                .iloc[0]
            )
            print(
                "max determinant cluster: ",
                max_det_cluster,
                ", action: ",
                max_det_action,
                ", sqrt determinant: ",
                "{:e}".format(max_det),
                ", count: ",
                max_det_count,
            )
        except ValueError:
            print("break")
            break

        try:
            ## finds the cluster-action group (that has enough observations) with the highest information radius (most incoherent) in order to split it into 2 clusters
            # print(grouped.query('Count >= @min_obs').sort_values('INFO_RADIUS', ascending=False))
            max_JSD_cluster, max_JSD_action, _, _, max_JSD, max_JSD_count = (
                grouped.query("Count >= @min_obs")
                .sort_values("INFO_RADIUS", ascending=False)
                .iloc[0]
            )
            print(
                "max JSD info radius cluster: ",
                max_JSD_cluster,
                ", action: ",
                max_JSD_action,
                ", JSD info radius: ",
                max_JSD,
                ", count: ",
                max_JSD_count,
            )
        except ValueError:
            print("break")
            break

        if incoherence_metric == "std":
            incoherences.append(max_std)
            incoherence_level = max_std
            cluster = max_std_cluster
            action = max_std_action
        elif incoherence_metric == "det":
            incoherences.append(max_det)
            incoherence_level = max_det
            cluster = max_det_cluster
            action = max_det_action
        elif incoherence_metric == "jsd":
            incoherences.append(max_JSD)
            incoherence_level = max_JSD
            cluster = max_JSD_cluster
            action = max_JSD_action

        ## sub-dataframe only keeping the rows from the most incoherent cluster-action group
        sub_df = df.query("CLUSTER == @cluster").query("ACTION == @action")

        # using the average between train_error and incoherence level
        # if 0.5*train_error + 0.5*std < min_error:
        #     min_error = 0.5*train_error + 0.5*std
        #     best_df = df
        #     opt_k = nc

        if incoherence_level >= threshold:
            nc += 1
            # print('incoherence above tolerated threshold')
            print("split iteration: ", n_iter)

            ## sub-dataframe of the rows from the cluster being split that DID NOT take the incoherent action
            rest_of_cluster = df.query("CLUSTER == @cluster").query("ACTION != @action")

            ## clustering sub_df (those observations in the cluster that took the incoherent action) based on the predicted probabilites of going to each next cluster and also the highest prob cluster (next cluster)

            kmeans = KMeans(n_clusters=2, random_state=0).fit(
                np.array(sub_df.iloc[:, (5 + p) :])
            )  # one of the features in the clustering is the next cluster
            labels = kmeans.labels_
            # print('Kmeans: ', np.mean(labels))

            ## reassigning observations in sub_df to their new clusters
            sub1 = sub_df[labels == 0]
            print("num in first cluster: ", len(sub1))
            sub2 = sub_df[labels == 1]
            print("num in other cluster: ", len(sub2))
            df.loc[sub1.index, "CLUSTER"] = counter + 1
            df.loc[sub2.index, "CLUSTER"] = counter + 2
            counter = max(df["CLUSTER"])  # new highest cluster number

            ## fits (binary) classifier to predict the newly assinged cluster from the features - trained only data from sub_df (the cluster-action group we split)
            model = _make_classifier(
                classification, split_classifier_params, spark_session
            )
            model.fit(df.loc[sub_df.index, features], df.loc[sub_df.index, "CLUSTER"])

            ## if the rest_of_cluster is not empty, assign the rest of the observations to the 2 new clusters using the trained binary classifier
            if len(rest_of_cluster) > 0:
                df.loc[rest_of_cluster.index, "CLUSTER"] = model.predict(
                    rest_of_cluster.loc[:, features]
                )

            df = df.sort_values(by=["ID", "TIME"], ascending=[True, True])
            df["NEXT_CLUSTER"] = df.groupby("ID")["CLUSTER"].shift(
                periods=-1, fill_value="None"
            )
            df["NEXT_CLUSTER"] = df.apply(
                lambda x: replace_None_next_cluster(x), axis=1
            )  # replaces None with 1111,2222,4444
            # clusters = list(set(df['CLUSTER']).union(set(df['NEXT_CLUSTER'])))

            le = preprocessing.LabelEncoder()
            clusters = list(set(df["CLUSTER"]).union(set(df["NEXT_CLUSTER"])))
            le.fit(clusters)
            df["CLUSTER"] = le.transform(df["CLUSTER"])
            # df["NEXT_CLUSTER"] = df["NEXT_CLUSTER"].astype(int)
            df["NEXT_CLUSTER"] = le.transform(df["NEXT_CLUSTER"])
            print(df["CLUSTER"].value_counts())

        model = _make_classifier(classification, split_classifier_params, spark_session)
        ## fits classifier to predict (probabilistically) the next cluster based on features and action
        X = df.loc[:, features + ["ACTION"]]
        y = df["NEXT_CLUSTER"].astype("int")
        model.fit(X, y)
        ## appends predicted probability columns for being in each cluster onto the df
        df = df.iloc[:, : (6 + p)].join(
            pd.DataFrame(model.predict_proba(df.loc[:, features + ["ACTION"]]))
        )  # why :6+p? Some problem with columns overlapping idk
        # print("df with predicted probabilites: ",df.isnull().any())

        ## for each cluster-action group, compute the sum of std (across whole group) in probability of going to each next cluster
        grouped = (
            df.groupby(["CLUSTER", "ACTION"])[df.columns[(6 + p) :]]
            .std()
            .sum(axis=1)
            .reset_index()
        )  # every column from 6+p to end is a predicted probability of going to a cluster
        grouped.columns = list(grouped.columns[:-1]) + [
            "Sum_std"
        ]  # renaming last column "Sum_std"

        # calculating train_error and incoherence level one last time
        train_error_1_step, R2_1_step = training_reward_error(
            df, end_state_rewards_df, P_method
        )
        train_error_whole = stochastic_training_value_error(
            df, end_state_rewards_df, P_method
        )
        test_error_1_step, test_R2_1_step = testing_reward_error(
            df, val_df, p, end_state_df, end_state_rewards_df, P_method
        )
        print(
            "num clusters: ",
            nc,
            "1 step value prediction R2: ",
            R2_1_step,
            "1 step validation R2: ",
            test_R2_1_step,
        )
        testing_error_1_step.append(test_error_1_step)
        training_error_whole.append(train_error_whole)
        training_error_1_step.append(train_error_1_step)
        training_R2_1_step.append(R2_1_step)
        testing_R2_1_step.append(test_R2_1_step)

        # grouped = pd.DataFrame(grouped)
        grouped["DET_INCOHERENCE"] = 0
        grouped["INFO_RADIUS"] = 0
        grouped["Count"] = (
            df.groupby(["CLUSTER", "ACTION"])["TIME"].count().reset_index()["TIME"]
        )
        for (
            ind
        ) in (
            grouped.index
        ):  # for each cluster, action pair, calculating the incoherence/dissimilarity in probability transitions
            cluster, action = grouped.loc[ind, ["CLUSTER", "ACTION"]]
            X = df.query("CLUSTER == @cluster and ACTION == @action")[
                df.columns[(6 + p) :]
            ]  # only the probability columns
            k = X.shape[0]
            info_radius_sum = 0
            avg_prob_vec = X.mean().values
            for i in range(k):
                info_radius_sum += JSD(X.values[i, :], avg_prob_vec)
            info_radius = info_radius_sum / k
            # print(Xmat.shape, "count: ", grouped.loc[ind, 'Count'])
            Xmat = X.to_numpy()
            det_incoherence = math.sqrt(
                abs(np.linalg.det(np.matmul(Xmat.transpose(), Xmat)))
            )
            # print(det_incoherence)
            grouped.loc[ind, "DET_INCOHERENCE"] = det_incoherence
            grouped.loc[ind, "INFO_RADIUS"] = info_radius

        # print(grouped)
        # print(grouped.query('Count >= @min_obs').sort_values('INFO_RADIUS', ascending=False))

        try:
            ## finds the cluster-action group (that has enough observations) with the highest summed std (most incoherent) in order to split it into 2 clusters
            # print(grouped.query('Count >= @min_obs').sort_values('Sum_std', ascending=False))
            max_std_cluster, max_std_action, max_std, _, _, max_std_count = (
                grouped.query("Count >= @min_obs")
                .sort_values("Sum_std", ascending=False)
                .iloc[0]
            )
            print(
                "max summed std cluster: ",
                max_std_cluster,
                ", action: ",
                max_std_action,
                ", summed std: ",
                max_std,
                ", count: ",
                max_std_count,
            )
        except ValueError:
            print("break")

        try:
            ## finds the cluster-action group (that has enough observations) with the highest determinant (most incoherent) in order to split it into 2 clusters
            # print(grouped.query('Count >= @min_obs').sort_values('DET_INCOHERENCE', ascending=False))
            max_det_cluster, max_det_action, _, max_det, _, max_det_count = (
                grouped.query("Count >= @min_obs")
                .sort_values("DET_INCOHERENCE", ascending=False)
                .iloc[0]
            )
            print(
                "max determinant cluster: ",
                max_det_cluster,
                ", action: ",
                max_det_action,
                ", sqrt determinant: ",
                "{:e}".format(max_det),
                ", count: ",
                max_det_count,
            )
        except ValueError:
            print("break")

        try:
            ## finds the cluster-action group (that has enough observations) with the highest information radius (most incoherent) in order to split it into 2 clusters
            # print(grouped.query('Count >= @min_obs').sort_values('INFO_RADIUS', ascending=False))
            max_JSD_cluster, max_JSD_action, _, _, max_JSD, max_JSD_count = (
                grouped.query("Count >= @min_obs")
                .sort_values("INFO_RADIUS", ascending=False)
                .iloc[0]
            )
            print(
                "max JSD info radius cluster: ",
                max_JSD_cluster,
                ", action: ",
                max_JSD_action,
                ", JSD info radius: ",
                max_JSD,
                ", count: ",
                max_JSD_count,
            )
        except ValueError:
            print("break")

        if incoherence_metric == "std":
            incoherences.append(max_std)
            incoherence_level = max_std
            cluster = max_std_cluster
            action = max_std_action
        elif incoherence_metric == "det":
            incoherences.append(max_det)
            incoherence_level = max_det
            cluster = max_det_cluster
            action = max_det_action
        elif incoherence_metric == "jsd":
            incoherences.append(max_JSD)
            incoherence_level = max_JSD
            cluster = max_JSD_cluster
            action = max_JSD_action

        print(
            "num clusters: ",
            nc,
            "incoherence_level: ",
            "{:e}".format(incoherence_level),
        )

        best_df = df

        # renumbering/encoding clusters nicely in df
        le = preprocessing.LabelEncoder()
        clusters = list(set(df["CLUSTER"]).union(set(df["NEXT_CLUSTER"])))
        le.fit(clusters)
        df["CLUSTER"] = le.transform(df["CLUSTER"])
        # df["NEXT_CLUSTER"] = df["NEXT_CLUSTER"].astype(int)
        df["NEXT_CLUSTER"] = le.transform(df["NEXT_CLUSTER"])
        print("number of points per cluster: ", df["CLUSTER"].value_counts())

        # renumbering/encoding clusters nicely in best_df
        le_best = preprocessing.LabelEncoder()
        # all final clusters will be present in the cluster column, except cluster "9999" which is in place of unknown data (last observed data point)
        # if this is not true, then append the list from "CLUSTER" and "NEXT_CLUSTER" together
        # clusters = best_df['CLUSTER'].unique()
        # clusters = np.append(clusters,[9999]) #we can't encode the next_cluster column without this line
        # 9999 should be encoded to the highest consectuive number, not left as 9999
        best_clusters = list(
            set(best_df["CLUSTER"]).union(set(best_df["NEXT_CLUSTER"]))
        )
        le_best.fit(best_clusters)
        best_df["CLUSTER"] = le_best.transform(best_df["CLUSTER"])
        best_df["NEXT_CLUSTER"] = best_df["NEXT_CLUSTER"].astype(int)
        best_df["NEXT_CLUSTER"] = le_best.transform(best_df["NEXT_CLUSTER"])

        df = df.sort_values(by=["ID", "TIME"], ascending=[True, True])
        # if best_df is None:
        #     raise ValueError("best_df is None.")
        best_df = best_df.sort_values(by=["ID", "TIME"], ascending=[True, True])
        #     plotting functions
        #     ## Plotting accuracy and value R2
        # print("training error length: ", len(training_error_whole))
        # print("incoherences length: ", len(incoherences))
        its = np.arange(init_k, nc + 1)
        # training_error = np.zeros(len(its)) + 100
        if plot:
            # if grid:

            if len(its) == len(incoherences):
                fig2, ax2 = plt.subplots()
                # norm_max = max(incoherences)
                ax2.plot(its, incoherences, label="incoherence")
                # ax2.plot(its, np.array(incoherences)/norm_max, label = "Max Incoherence")
                # ax2.plot(its, np.array(thresholds)/norm_max, 'r-', label = "Threshold")
                # if testing:
                #     ax2.plot(its, testing_error, label = "Testing Error")
                # if n>0:
                #     ax2.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
                # ax2.set_ylim(0)
                ax2.set_xlabel("# of Clusters")
                ax2.set_ylabel("Max incoherence level")
                ax2.set_title("Max incoherence by number of clusters")
                ax2.legend()
                plt.show()

            if len(its) == len(training_R2_1_step):
                fig3, ax3 = plt.subplots()
                ax3.plot(
                    its, training_R2_1_step, label="Training 1step Value Prediction R2"
                )
                # ax3.set_ylim(0)
                ax3.set_xlabel("# of Clusters")
                ax3.set_ylabel("Value prediction R2")
                ax3.set_title("Value prediction R2 by number of clusters")
                ax3.legend()
                fig6, ax6 = plt.subplots()
                # norm_max = max(incoherences)
                ax6.plot(its, testing_R2_1_step, label="testing 1 step prediction R2")
                # ax6.set_ylim(0)
                ax6.set_xlabel("# of Clusters")
                ax6.set_ylabel("Testing value prediction R2")
                ax6.set_title("Testing value prediction R2 by number of clusters")
                ax6.legend()
                plt.show()

            if len(its) == len(training_error_1_step):
                fig1, ax1 = plt.subplots()
                # ax1.plot(its, training_R2, label= "Training R2")
                ax1.plot(
                    its,
                    training_error_1_step,
                    label="Training 1step Value Prediction Error",
                )
                # if testing:
                #     ax1.plot(its, testing_acc, label = "Testing Accuracy")
                # ax1.plot(its, testing_R2, label = "Testing R2")
                # if n>0:
                #     ax1.axvline(x=n,linestyle='--',color='r') #Plotting vertical line at #cluster =n
                # ax1.set_ylim(0)
                ax1.set_xlabel("# of Clusters")
                ax1.set_ylabel("1step value prediction error")
                ax1.legend()

            if len(its) == len(testing_error_1_step):
                fig5, ax5 = plt.subplots()
                ax5.plot(
                    its, testing_error_1_step, label="testing 1 step prediction error"
                )
                # ax5.set_ylim(0)
                ax5.set_xlabel("# of Clusters")
                ax5.set_ylabel("Testing value prediction error")
                ax5.set_title("Testing value prediction error by number of clusters")
                ax5.legend()
                plt.show()

            if len(its) == len(training_error_whole):
                fig4, ax4 = plt.subplots()
                ax4.plot(
                    its, training_error_whole, label="Training Value Prediction Error"
                )
                # ax4.set_ylim(0)
                ax4.set_xlabel("# of Clusters")
                ax4.set_ylabel("Training value prediction error")
                ax4.set_title("Training Value prediction error by number of clusters")
                ax4.legend()

        try:
            df_train_error = pd.DataFrame(
                list(zip(its, training_error_1_step)), columns=["Clusters", "Error"]
            )

        except ValueError:
            print("lengths match?", len(its) == len(incoherences))

        try:
            df_incoherences = pd.DataFrame(
                list(zip(its, incoherences)), columns=["Clusters", "Incoherence Level"]
            )
        except ValueError:
            print("lengths match?", len(its) == len(incoherences))
        #     if testing:
        #         df_test_error = pd.DataFrame(list(zip(its, testing_error)), \
        #                                   columns = ['Clusters', 'Error'])
        #         return (df_new, df_incoherences, df_train_error,df_test_error, best_df, opt_k, split_scores)
        # =============================================================================
        # the output dataframe has updated CLUSTER and NEXT_CLUSTER columns as well as predicted probabilities of ending up in each of the clusters next
        return df, best_df, df_train_error, df_incoherences, opt_k

    # initializing lists for error & accuracy data
    #     training_R2 = []
    #     testing_R2 = []
    #     training_acc = []
    #     testing_acc = []
    #     testing_error = []
    training_error = []
    prescriptive_error = []
    #
    #     incoherences = []
    #     split_scores = []
    #     thresholds = []

    k = int(df["CLUSTER"].nunique())  # initial number of clusters
    nc = k  # number of clusters

    # df_new = deepcopy(df)

    # storing optimal df
    best_df = None
    opt_k = None
    min_error = float("inf")

    # backup values in case threshold fails

    #     backup_min_error = float('inf')
    #     backup_df = None
    #     backup_opt_k = None

    df = df.copy()
    if df is None:
        print("Uh oh")
    else:
        print(df.columns)
    features = list(
        set(df.columns).difference(
            set(["ID", "TIME", "RISK", "ACTION", "CLUSTER", "NEXT_CLUSTER"])
        )
    )
    print("correct number of features", p == len(features))

    # assumes number of columns besides features is 6 (ID, TIME, ACTION, RISK, CLUSTER, NEXT_CLUSTER)
    std = 1000  # initiate with a high standard deviation
    counter = max(df["CLUSTER"])  # highest cluster number
    n_iter = 0
    while (
        std >= threshold and n_iter <= max_iter
    ):  # while the standard deviation is still high and we haven't split the max number of times yet
        n_iter += 1
        model = RandomForestClassifier()
        ## fits classifier to predict (probabilistically) the next cluster based on features and action
        X = df.loc[:, features + ["ACTION"]]
        y = df["NEXT_CLUSTER"].astype("int")
        model.fit(X, y)
        ## appends predicted probability columns for being in each cluster onto the df
        df = df.iloc[:, : (6 + p)].join(
            pd.DataFrame(model.predict_proba(df.loc[:, features + ["ACTION"]]))
        )  # why :6+p? Some problem with columns overlapping idk

        ## for each cluster-action group, compute the sum of std (across whole group) in probability of going to each next cluster
        grouped = (
            df.groupby(["CLUSTER", "ACTION"])[df.columns[(6 + p) :]]
            .std()
            .sum(axis=1)
            .reset_index()
        )  # every column from 6+p to end is a predicted probability of going to a cluster
        grouped.columns = list(grouped.columns[:-1]) + [
            "Sum_std"
        ]  # renaming last column "Sum_std"

        ## counting the number of observations in each cluster-action group
        grouped["Count"] = (
            df.groupby(["CLUSTER", "ACTION"])["TIME"].count().reset_index()["TIME"]
        )
        try:
            ## finds the cluster-action group (that has enough observations) with the highest summed std (most incoherent) in order to split it into 2 clusters
            cluster, action, std, _ = (
                grouped.query("Count >= @min_obs")
                .sort_values("Sum_std", ascending=False)
                .iloc[0]
            )
            print("cluster: ", cluster, "action: ", action)
        except ValueError:
            print("break")
            break

        ## sub-dataframe only keeping the rows from the most incoherent cluster-action group
        sub_df = df.query("CLUSTER == @cluster").query("ACTION == @action")

        print("Summed std: ", std)
        if std >= threshold:
            nc += 1
            print("incoherence above tolerated threshold")
            print("split iteration: ", n_iter)

            ## sub-dataframe of the rows from the cluster being split that DID NOT take the incoherent action
            rest_of_cluster = df.query("CLUSTER == @cluster").query("ACTION != @action")

            ## clustering sub_df (those observations in the cluster that took the incoherent action) based on the predicted probabilites of going to each next cluster and also the highest prob cluster (next cluster)
            kmeans = KMeans(n_clusters=2, random_state=0).fit(
                np.array(sub_df.iloc[:, (5 + p) :])
            )  # one of the features in the clustering is the next cluster
            labels = kmeans.labels_
            # print('Kmeans: ', np.mean(labels))

            ## reassigning observations in sub_df to their new clusters
            sub1 = sub_df[labels == 0]
            sub2 = sub_df[labels == 1]
            df.loc[sub1.index, "CLUSTER"] = counter + 1
            df.loc[sub2.index, "CLUSTER"] = counter + 2
            counter = max(df["CLUSTER"])  # new highest cluster number

            ## fits (binary) classifier to predict the newly assinged cluster from the features - trained only data from sub_df (the cluster-action group we split)
            model = RandomForestClassifier()
            model.fit(df.loc[sub_df.index, features], df.loc[sub_df.index, "CLUSTER"])

            ## if the rest_of_cluster is not empty, assign the rest of the observations to the 2 new clusters using the trained binary classifier
            if len(rest_of_cluster) > 0:
                df.loc[rest_of_cluster.index, "CLUSTER"] = model.predict(
                    rest_of_cluster.loc[:, features]
                )

            df = df.sort_values(by=["ID", "TIME"], ascending=[True, True])
            df["NEXT_CLUSTER"] = df.groupby("ID")["CLUSTER"].shift(
                periods=-1, fill_value="None"
            )
            df["NEXT_CLUSTER"] = df.apply(
                lambda x: replace_None_next_cluster(x), axis=1
            )

            le = preprocessing.LabelEncoder()
            
            clusters = list(set(df["CLUSTER"]).union(set(df["NEXT_CLUSTER"])))
            le.fit(clusters)
            df["CLUSTER"] = le.transform(df["CLUSTER"])
            # df["NEXT_CLUSTER"] = df["NEXT_CLUSTER"].astype(int)
            df["NEXT_CLUSTER"] = le.transform(df["NEXT_CLUSTER"])

            end_state_rewards_df = end_state_df.groupby(["end_state"])["Reward"].mean()
            end_state_rewards_df = (
                end_state_rewards_df.reset_index()
            )  # makes both end_state and Reward into columns again
            self.df_trained = df.copy()
            P, R = get_MDP_stochastic(self.df_trained, end_state_rewards_df, P_method)
            self.P = P
            self.P_df = pd.DataFrame(P)
            self.R_df = R
            self.R = R
            # store next_clusters dataframe
            self.nc = next_clusters(self.df_trained)  # adds 'purity' and 'count' coluns

            if MDP_solver == "Robust":
                v, pi, Vals = solve_MDP_Robust(
                    self,
                    alpha=0.2,  # statistical alpha threshold
                    beta=0.6,  # statistical beta threshold
                    min_action_obs=-1,  # int: least number of actions that must be seen
                    min_action_purity=0.3,  # float: percentage purity above which is acceptable
                    prob=MDP_obj,  # str: 'max', or 'min' for maximization or minimization problem
                    gamma=gamma,  # discount factor
                    epsilon=10 ** (-10),
                    p=True,
                    prob_thresh=0,
                )
            elif MDP_solver == "Robust expected":
                v, pi, Vals = solve_MDP_Robust_expected(
                    self,
                    alpha=0.2,  # statistical alpha threshold
                    beta=0.6,  # statistical beta threshold
                    min_action_obs=-1,  # int: least number of actions that must be seen
                    min_action_purity=0.3,  # float: percentage purity above which is acceptable
                    prob=MDP_obj,  # str: 'max', or 'min' for maximization or minimization problem
                    gamma=gamma,  # discount factor
                    epsilon=10 ** (-10),
                    p=True,
                    prob_thresh=0,
                )
            elif MDP_solver == "regular":
                v, pi, Vals = solve_MDP(
                    self,
                    alpha=0.2,  # statistical alpha threshold
                    beta=0.6,  # statistical beta threshold
                    min_action_obs=-1,  # int: least number of actions that must be seen
                    min_action_purity=0.3,  # float: percentage purity above which is acceptable
                    prob=MDP_obj,  # str: 'max', or 'min' for maximization or minimization problem
                    gamma=gamma,  # discount factor
                    epsilon=10 ** (-10),
                    p=True,
                )

            prescrip_error = np.mean(v[0:nc])  # mean for just the real clusters
            prescriptive_error.append(prescrip_error)

            train_error = stochastic_training_value_error(
                df,
                end_state_rewards_df,
                P_method=1,
                gamma=gamma,
                relative=False,
                h=-1,
                num_trajectories=1,
            )  # num_trajectories is how many random stochastic trajectories it generates when calculating the training value estimation error
            training_error.append(train_error)

            print("value prediction error", train_error)
            print("average value", prescrip_error)
            print(
                "weighted total",
                pred_weight * train_error + prescrip_weight * prescrip_error,
            )
            if pred_weight * train_error + prescrip_weight * prescrip_error < min_error:
                min_error = pred_weight * train_error + prescrip_weight * prescrip_error
                best_df = df
                opt_k = nc

    # renumbering/encoding clusters nicely in df
    le = preprocessing.LabelEncoder()
    clusters = list(set(df["CLUSTER"]).union(set(df["NEXT_CLUSTER"])))
    le.fit(clusters)
    df["CLUSTER"] = le.transform(df["CLUSTER"])
    # df["NEXT_CLUSTER"] = df["NEXT_CLUSTER"].astype(int)
    df["NEXT_CLUSTER"] = le.transform(df["NEXT_CLUSTER"])

    if best_df != None:
        # renumbering/encoding clusters nicely in best_df
        le_best = preprocessing.LabelEncoder()
        clusters = list(set(best_df["CLUSTER"]).union(set(best_df["NEXT_CLUSTER"])))
        le_best.fit(clusters)
        best_df["CLUSTER"] = le_best.transform(best_df["CLUSTER"])
        best_df["NEXT_CLUSTER"] = best_df["NEXT_CLUSTER"].astype(int)
        best_df["NEXT_CLUSTER"] = le_best.transform(best_df["NEXT_CLUSTER"])

    df = df.sort_values(by=["ID", "TIME"], ascending=[True, True])
    if best_df is None:
        raise ValueError("Best df is none?")
    best_df = best_df.sort_values(by=["ID", "TIME"], ascending=[True, True])
    #     plotting functions
    #     ## Plotting accuracy and value R2
    its = np.arange(k + 1, nc + 1)
    df_train_error = pd.DataFrame(
        list(zip(its, training_error, prescriptive_error)),
        columns=["Clusters", "value prediction error", "average predicted value"],
    )
    return df, best_df, df_train_error, opt_k


#################################################################


# Splitter algorithm with Group K-fold cross-validation (number of folds from param cv)
# Returns dataframes of incoherences, errors, and splitter split-scores; these
# can be used to determine optimal clustering.
def fit_CV(
    df,
    pfeatures,
    th,
    clustering,
    distance_threshold,
    eta,
    precision_thresh,
    classification,
    split_classifier_params,
    max_k,
    n_clusters,
    random_state,
    h,
    gamma=1,
    verbose=False,
    cv=5,
    n=-1,
    plot=False,
):
    df_training_error = pd.DataFrame(columns=["Clusters"])
    df_testing_error = pd.DataFrame(columns=["Clusters"])
    df_incoherences = pd.DataFrame(columns=["Clusters"])

    gkf = GroupKFold(n_splits=cv)
    # shuffle the ID's (create a new column), and do splits based on new ID's
    random.seed(datetime.now())
    g = [df for _, df in df.groupby("ID")]
    random.shuffle(g)
    df = pd.concat(g).reset_index(drop=True)
    ids = df.groupby(["ID"], sort=False).ngroup()
    df["ID_shuffle"] = ids

    for train_idx, test_idx in gkf.split(df, y=None, groups=df["ID_shuffle"]):
        df_train = df[df.index.isin(train_idx)]
        df_test = df[df.index.isin(test_idx)]
        # print('IDs in testing', df_test['ID'].unique())
        #################################################################
        # Initialize Clusters
        df_init = initializeClusters(
            df_train,
            clustering=clustering,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            random_state=random_state,
        )
        # k = df_init['CLUSTER'].nunique()
        # print('k', k)
        # print(df_init)
        #################################################################

        #################################################################
        # Run Iterative Learning Algorithm

        (
            df_new,
            incoherences,
            training_error,
            testing_error,
            best_df,
            opt_k,
            split_scores,
        ) = splitter(
            df_init,
            pfeatures,
            th,
            eta=eta,
            precision_thresh=precision_thresh,
            df_test=df_test,
            testing=True,
            max_k=max_k,
            classification=classification,
            split_classifier_params=split_classifier_params,
            h=h,
            gamma=gamma,
            verbose=False,
            n=n,
            plot=plot,
        )

        df_training_error = df_training_error.merge(
            training_error, how="outer", on=["Clusters"]
        )
        df_testing_error = df_testing_error.merge(
            testing_error, how="outer", on=["Clusters"]
        )
        df_incoherences = df_incoherences.merge(
            incoherences, how="outer", on=["Clusters"]
        )

    df_training_error.set_index("Clusters", inplace=True)
    df_testing_error.set_index("Clusters", inplace=True)
    df_incoherences.set_index("Clusters", inplace=True)

    df_training_error.dropna(inplace=True)
    df_testing_error.dropna(inplace=True)
    df_incoherences.dropna(inplace=True)

    cv_training_error = np.mean(df_training_error, axis=1)
    cv_testing_error = np.mean(df_testing_error, axis=1)
    cv_incoherences = np.mean(df_incoherences, axis=1)

    if plot:
        fig1, ax1 = plt.subplots()
        # its = np.arange(k+1,k+1+len(cv_training_error))
        ax1.plot(
            cv_training_error.index.values, cv_training_error, label="CV Training Error"
        )
        # ax1.plot(its, cv_testing_error, label = "CV Testing Error")
        ax1.plot(
            cv_testing_error.index.values, cv_testing_error, label="CV Testing Error"
        )
        # ax1.plot(its, training_acc, label = "Training Accuracy")
        # ax1.plot(its, testing_acc, label = "Testing Accuracy")
        if n > 0:
            ax1.axvline(
                x=n, linestyle="--", color="r"
            )  # Plotting vertical line at #cluster =n
        ax1.set_ylim(0)
        ax1.set_xlabel("# of Clusters")
        ax1.set_ylabel("Mean CV Error or Accuracy %")
        ax1.set_title("Mean CV Error and Accuracy During Splitting")
        ax1.legend()

    return (cv_incoherences, cv_training_error, cv_testing_error, split_scores)
