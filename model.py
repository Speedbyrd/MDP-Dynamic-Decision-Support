#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:47:03 2020

Model Class that runs the MRL algorithm on any data.

@author: janiceyang
"""

#################################################################
# Load Libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from .clustering import (
    fit_CV,
    initializeClusters,
    splitter,
    splitter_stochastic,
)
from .testing import (
    predict_cluster,
    training_value_error,
    get_MDP_stochastic,
    predict_value_of_cluster,
    testing_value_error,
    model_trajectory,
    next_clusters,
)
from .MDPToolsRobust import (
    SolveMDP,
    SolveMDP_Robust,
    SolveMDP_Robust_expected,
    SolveMDP_Robust_quantiles,
    SolveMDP_ICTE,
)
from sklearn.metrics import accuracy_score
from scipy.stats import binom

#################################################################


class MDP_model:
    def __init__(self):
        self.df = None  # original dataframe from data
        self.pfeatures = None  # number of features
        self.CV_error = None  # error at minimum point of CV
        self.CV_error_all = None  # errors of different clusters after CV
        self.training_error = None  # training errors after last split sequence
        self.incoherences = None  # summed std for stochastic
        self.split_scores = None  # cv error from splitter (if GridSearch used)
        self.opt_k = None  # number of clusters in optimal clustering
        self.eta = None  # incoherence threshold
        self.df_trained = None  # dataframe after optimal training
        self.df_trained_stratified = None
        self.best_df_trained = None
        self.m = None  # model for predicting cluster number from features #CHANGE NAME
        self.clus_pred_accuracy = (
            None  # accuracy score of the cluster prediction function
        )
        self.P_df = None  # Transition function of the learnt MDP, includes sink node if end state exists
        self.P_df_stratified = None
        self.P_df_1 = None
        self.P_df_2 = None
        self.R_df = None  # Reward function of the learnt MDP, includes sink node of reward 0 if end state exists
        self.R_df_stratified = None
        self.nc = None  # dataframe similar to P_df, but also includes 'count' and 'purity' cols
        self.nc_stratified = None
        self.v = None  # value after MDP solved
        self.pi = None  # policy after MDP solved
        self.v_stratified = None  # value after MDP solved
        self.pi_stratified = None  # policy after MDP solved
        self.P = None  # P_df but in matrix form of P[a, s, s'], with alterations
        # where transitions that do not pass the action and purity thresholds
        # now lead to a new cluster with high negative reward
        self.P_stratified = None
        self.R = None  # R_df but in matrix form of R[a, s]
        self.R_stratified = None
        self.stratifying_label_encoder = None
        self.stratifying_le_classes = None

    # fit_CV() takes in parameters for prediction, and trains the model on the
    # optimal clustering for a given horizon h (# of actions), using cross
    # validation. See fit_CV in clustering.py for further documentation.
    def fit_CV(
        self,
        data,  # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
        # Needs a dataframe where 'ACTION' == 'None' if goal state is reached.
        pfeatures,  # int: number of features
        h=5,  # int: time horizon (# of actions we want to optimize)
        gamma=1,  # discount value
        max_k=70,  # int: max number of clusters
        distance_threshold=0.05,  # clustering diameter for Agglomerative clustering
        cv=5,  # number of folds for cross validation
        th=0,  # splitting threshold
        eta=float(
            "inf"
        ),  # incoherence threshold, calculated by eta*sqrt(datapoints)/clusters
        precision_thresh=1e-14,  # precision threshold
        classification="DecisionTreeClassifier",  # classification method
        split_classifier_params={"random_state": 0},
        clustering="Agglomerative",  # clustering method from Agglomerative, KMeans, and Birch
        n_clusters=None,  # number of clusters for KMeans
        random_state=0,
        plot=False,
        verbose=False,
    ):
        df = data.copy()

        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        self.eta = eta

        # run cross validation on the data to find best clusters
        cv_incoherences, cv_training_error, cv_testing_error, split_scores = fit_CV(
            self.df,
            self.pfeatures,
            th=th,
            clustering=clustering,
            distance_threshold=distance_threshold,
            eta=eta,
            precision_thresh=precision_thresh,
            classification=classification,
            split_classifier_params=split_classifier_params,
            max_k=max_k,
            n_clusters=n_clusters,
            random_state=random_state,
            h=h,
            gamma=gamma,
            verbose=verbose,
            cv=cv,
            n=-1,
            plot=plot,
        )

        # store cv testing error
        cv_testing_error = pd.concat(
            [
                cv_testing_error.rename("Testing Error"),
                cv_training_error.rename("Training Error"),
                cv_incoherences.rename("Incoherence"),
            ],
            axis=1,
        )
        self.CV_error_all = cv_testing_error

        cv_testing_error.reset_index(inplace=True)

        # find optimal cluster after filtering for eta
        inc_thresh = self.eta * self.df.shape[0] ** 0.5
        filtered = cv_testing_error.loc[
            cv_testing_error["Incoherence"]
            < inc_thresh / (cv_testing_error["Clusters"])
        ]

        filtered.set_index("Clusters", inplace=True)
        k = filtered["Testing Error"].idxmin()

        if verbose:
            print("CV Testing Error")
            print(cv_testing_error)
            print("best clusters:", k)
        self.opt_k = k

        # actual training on all the data
        df_init = initializeClusters(
            self.df,
            clustering=clustering,
            n_clusters=n_clusters,
            distance_threshold=distance_threshold,
            random_state=random_state,
        )

        print(df_init)
        # Rename end state to 'end'
        df_init.loc[df_init["ACTION"] == "None", "NEXT_CLUSTER"] = "End"

        (
            df_new,
            df_incoherences,
            training_error,
            testing_error,
            best_df,
            opt_k,
            split_scores,
        ) = splitter(
            df_init,
            pfeatures=self.pfeatures,
            th=th,
            eta=self.eta,
            precision_thresh=precision_thresh,
            df_test=None,
            testing=False,
            max_k=self.opt_k,
            classification=classification,
            split_classifier_params=split_classifier_params,
            h=h,
            gamma=gamma,
            verbose=verbose,
            plot=plot,
        )

        # storing trained dataset and predict_cluster function and accuracy
        self.df_trained = df_new
        self.m = predict_cluster(df_new, self.pfeatures)
        pred = self.m.predict(df_new.iloc[:, 2 : 2 + self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, df_new["CLUSTER"])

        # store final training error and incoherenes
        self.training_error = training_value_error(self.df_trained)
        self.incoherences = df_incoherences
        self.split_scores = split_scores

        # store P_df and R_df values
        P_df, R_df = get_MDP(self.df_trained)
        self.P_df = P_df
        self.R_df = R_df

        # store next_clusters dataframe
        self.nc = next_clusters(df_new)

    # fit() takes in the parameters for prediction, and directly fits the model
    # to the data without running cross validation. If optimize is set to True,
    # stores the best clustering in self.df_trained; otherwise stores the
    # clustering at when max_k number of clusters is reached.
    def fit_stochastic(
        self,
        # stratified,
        data,  # df: dataframe in the format ['ID', 'TIME', ...features..., 'RISK', 'ACTION']
        # Needs a dataframe where 'ACTION' == 'None' if goal state is reached.
        val_df,
        end_state_df,
        # end_state_list,
        pfeatures,  # int: number of features
        P_method=1,
        incoherence_threshold=0,  # maximum sum of standard deviation in transition probabilities tolerable
        h=-1,  # int: time horizon (# of actions we want to optimize)
        gamma=1,  # discount value
        max_k=70,  # int: max number of clusters
        distance_threshold=0.05,  # clustering diameter for Agglomerative clustering
        cv=5,  # number for cross validation
        th=0,  # splitting threshold
        eta=float("inf"),  # incoherence threshold
        precision_thresh=1e-14,  # precision threshold
        classification="RandomForestClassifier",  # classification method
        split_classifier_params={"random_state": 0},  # dict of classifier params
        clustering="Agglomerative",  # clustering method from Agglomerative, KMeans, and Birch
        n_clusters=None,  # number of initial clusters
        random_state=0,
        plot=False,
        optimize=True,
        verbose=False,
        df_init=None,
        min_obs=2000,
        incoherence_metric="std",
        spark_session=None,  # When provided, uses Spark MLlib RF for distributed training
    ):
        df = data.copy()

        # save relevant data
        self.df = df
        self.pfeatures = pfeatures
        self.eta = eta
        self.t_max = df["TIME"].max()
        self.r_max = abs(df["RISK"]).max()

        # training on all the data

        if df_init.empty:
            df_init = initializeClusters(
                self.df,
                end_state_df,
                clustering=clustering,
                n_clusters=n_clusters,
                distance_threshold=distance_threshold,
                random_state=random_state,
            )

        df, best_df, df_train_error, df_incoherences, opt_k = splitter_stochastic(
            df_init,  # pandas dataFrame #with 'ClUSTER' and 'NEXT_CLUSTER columns
            val_df,
            end_state_df,
            # end_state_list,
            p=pfeatures,  # integer: number of features
            threshold=incoherence_threshold,  # threshold for minimum standard deviation to split
            P_method=P_method,
            eta=25,  # not used
            precision_thresh=1e-14,  # precision threshold when considering new min value error
            df_test=None,  # df_test provided for cross validation
            testing=False,  # True if we are cross validating
            max_iter=(max_k - n_clusters) - 1,  # int: max number of cluster splits
            classification="RandomForestClassifier",  # string: classification alg
            split_classifier_params={
                "random_state": 0
            },  # dict: classification params #not used
            h=-1,  # idk what this is, not used
            gamma=gamma,  # idk what this is, not used
            verbose=False,  # idk what this is, not used
            n_param=-1,  # idk what this is, not used
            plot=plot,
            min_obs=min_obs,
            incoherence_metric=incoherence_metric,
            spark_session=spark_session,
        )

        #         # store all training errors
        self.training_error = df_train_error
        self.incoherences = df_incoherences
        #         self.split_scores = split_scores

        # storing trained dataset and predict_cluster function, depending on
        # whether optimization was selected
        # incoherence and precision thresholds were already applied
        # within splitter to find best_df and opt_k

        self.opt_k = opt_k
        self.best_df_trained = best_df  # .iloc[:,:(6+pfeatures)]

        if optimize:
            # self.df_trained = best_df.iloc[:,:(6+pfeatures)] #cuts off the predicted probability columns
            self.df_trained = best_df
            # k = self.training_error['Clusters'].iloc[self.training_error['Error'].idxmin()]
        else:
            # self.df_trained = df.iloc[:,:(6+pfeatures)]
            self.df_trained = df

        end_state_rewards_df = end_state_df.groupby(["end_state"])["Reward"].mean()
        end_state_rewards_df = (
            end_state_rewards_df.reset_index()
        )  # makes both end_state and Reward into columns again
        # end_state_rewards_dict = dict()
        # for row in range(len(end_state_rewards_df)):
        #     end_state_rewards_dict[
        #         end_state_rewards_df.loc[row, "end_state"]
        #     ] = end_state_rewards_df.loc[row, "Reward"]

        P2, R = get_MDP_stochastic(self.df_trained, end_state_rewards_df, 2)
        # self.P = P #P has 3 indices: CLUSTER, ACTION, and NEXT_CLUSTER, and one column called 0
        self.P_df_2 = pd.DataFrame(P2)

        P1, R = get_MDP_stochastic(self.df_trained, end_state_rewards_df, 1)
        # self.P = P #P has 3 indices: CLUSTER, ACTION, and NEXT_CLUSTER, and one column called 0
        # start changes 2/14
        # P1_long = P1.rename("prob").reset_index()

        # P1_mat = (
        #     P1_long.pivot_table(
        #         index=["ACTION", "CLUSTER"],
        #         columns="NEXT_CLUSTER",
        #         values="prob",
        #         fill_value=0.0,
        #         aggfunc="sum",
        #     )
        #     .reset_index()
        # )

        # end changes 2/14

        # start changes 2/15
        self.P_df_1 = P1.reorder_levels(
            ["ACTION", "CLUSTER", "NEXT_CLUSTER"]
        ).sort_index()
        # end changes 2/15

        # non-stratified stuff
        # death_state = self.df_trained['NEXT_CLUSTER'].max()
        # sink_state = death_state + 1
        # readmit_state = death_state - 1
        # discharge_state = death_state - 2
        # store P_df and R_df values
        P, R = get_MDP_stochastic(self.df_trained, end_state_rewards_df, P_method)
        # self.P = P #P has 3 indices: CLUSTER, ACTION, and NEXT_CLUSTER, and one column called 0
        self.P_df = pd.DataFrame(P)
        self.R_df = R
        self.R = R
        self.m = predict_cluster(self.df_trained, self.pfeatures)
        pred = self.m.predict(self.df_trained.iloc[:, 2 : 2 + self.pfeatures])
        self.clus_pred_accuracy = accuracy_score(pred, self.df_trained["CLUSTER"])
        # store next_clusters dataframe
        self.nc = next_clusters(self.df_trained)  # adds 'purity' and 'count' columns

    # predict() takes a list of features and a time horizon, and returns
    # the predicted value after all actions are taken in order
    def predict(
        self, features, actions  # list: list OR array of features
    ):  # list: list of actions
        # predict initial cluster
        s = int(self.m.predict([features]))

        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df, self.R_df, s, actions)
        return v

    # predict_forward() takes an ID & actions, and returns the predicted value
    # for this ID after all actions are taken in order
    def predict_forward(self, ID, actions):
        # cluster of this last point
        s = self.df_trained[self.df_trained["ID"] == ID].iloc[-1, -2]

        # predict value sum given starting cluster and action path
        v = predict_value_of_cluster(self.P_df, self.R_df, s, actions)
        return v

    # testing_error() takes a df_test, then computes and returns the testing
    # error on this trained model
    def testing_error(self, df_test, relative=False, h=-1):
        error = testing_value_error(
            df_test, self.df_trained, self.m, self.pfeatures, relative=relative, h=h
        )

        return error

    # solve_MDP() takes the trained model as well as parameters for gamma,
    # epsilon, whether the problem is a minimization or maximization one,
    # and the threshold cutoffs to not include actions that don't appear enough
    # in each state, as well as purity cutoff for next_states that do not
    # represent enough percentage of all the potential next_states,
    # and returns the the value and policy. When solving the MDP, creates an
    # artificial punishment state that is reached for state/action pairs that
    # don't meet the above cutoffs; also creates a sink node of reward 0
    # after the goal state or punishment state is reached.
    def solve_MDP(
        self,
        removed_actions=[],
        # alpha = 0.2, # statistical alpha threshold
        # beta = 0.6, # statistical beta threshold
        # min_action_obs = -1, # int: least number of actions that must be seen
        # min_action_purity = 0.3, # float: percentage purity above which is acceptable
        prob="min",  # str: 'max', or 'min' for maximization or minimization problem
        gamma=0.9,  # discount factor
        epsilon=10 ** (-10),
        print_sol=True,
    ):
        # all of this filtering stuff is for the deterministic MDP algo.
        # if default value, then scale the min threshold with data size, ratio 0.008
        # if min_action_obs == -1:
        #     min_action_obs = max(5, 0.008*self.df_trained.shape[0])

        R = self.R_df.copy()
        P_df = self.P_df.copy()
        P_df_1 = P_df.copy()
        # P_df['count'] = self.nc['count']
        # P_df['purity'] = self.nc['purity']
        P_df = P_df.reset_index()

        # record parameters of transition dataframe
        num_a = P_df["ACTION"].nunique()
        num_s = len(
            list(set(P_df["CLUSTER"]).union(set(P_df["NEXT_CLUSTER"])))
        )  # number of clusters, already includes the sink node#P_df['CLUSTER'].nunique() #
        actions = list(P_df["ACTION"].unique())

        # # Take out rows that don't pass statistical alpha test
        # P_alph = P_df.loc[(1-binom.cdf(P_df['purity']*(P_df['count']), P_df['count'],\
        #                               beta))<=alpha]

        # # Take out rows where actions or purity below threshold
        # P_thresh = P_alph.loc[(P_alph['count']>min_action_obs)&(P_alph['purity']>min_action_purity)]

        print("incomplete clusters and missing actions")
        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(
            P_df.groupby("CLUSTER")["ACTION"].nunique() < num_a
        )[
            0
        ]  # now it does what it's supposed to with nunique
        # stores tuples of clusters and missing action
        print(incomplete_clusters)
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(
                actions, P_df.loc[P_df["CLUSTER"] == c]["ACTION"].unique()
            )
            print("cluster: ", c, "missing actions: ", not_present, len(not_present))
            for a in not_present:
                missing_pairs.append((c, a))
        # print(missing_pairs)
        # print(len(missing_pairs))
        print("---------------------------------------------------------------")

        P = np.zeros((num_a, num_s + 1, num_s + 1))
        for index in P_df_1.index:
            P[index[1], index[0], index[2]] = P_df_1.loc[
                index
            ]  # action, cluster, next_cluster

        # punishment node = num_s
        # print('current R indices: ', R.index, 'new sink node: ', num_s-1, 'new punishment state: ', num_s)
        if prob == "max":
            R.loc[num_s] = -100000000
        if prob == "min":
            R.loc[num_s] = (
                100000000  # s should be the name of the punishment node and have a reward of infinity
            )

        for a in removed_actions:
            P[a, :, :] = 0
            P[a, :, -1] = 1

        # reinsert transition for missing cluster-action pairs (goes to punishment node)
        # print("the missing pairs transition to the punishment node")
        for pair in missing_pairs:
            c, a = pair
            P[a, c, -1] = 1
            # print(pair)
            # print(P[a,c,num_s])
        # print("-------------------------------------------------------")

        # punishment node to 0 reward sink (if sink was created in get_MDP):
        # if 'End' in self.df_trained['NEXT_CLUSTER'].unique():
        # print("For every action, punishment node transitions to sink node")
        for a in range(num_a):
            P[a, -1, -2] = 1  # punishment node transitions to sink node?
            # print(a, P[a,num_s,num_s-1])
        # print("-----------------------------------------------------------")

        # PERHAPS THIS SHOULD BE AFTER EVERYTHING
        # basically doing the same thing as reinsert missing cluster-action pairs below
        # quick bandaid fix for probabilities not summing to 1
        print("any cluster action pair whose transition probabilites don't sum to one")
        for a in range(P.shape[0]):
            for c in range(P.shape[1]):
                if sum(P[a, c, :]) < 0.99:
                    print(c, a, sum(P[a, c, :]))
                    P[a, c, :] = 0
                    P[a, c, -1] = (
                        1  # for actions we don't know what happens next, send them to the punishment node
                    )
        print("------------------------------------------------------------")

        T_max = self.df_trained["TIME"].max()
        r_max = abs(self.df_trained["RISK"]).max()
        self.t_max = T_max
        self.r_max = r_max
        # for i in range(a): i genuinely do not know what this does
        #     if prob == 'max':
        #         # take T-max * max(abs(reward)) * 2
        #         R.append(np.append(np.array(self.R_df),-self.t_max*self.r_max*2))
        #     else:
        #         R.append(np.append(np.array(self.R_df),self.t_max*self.r_max*2))
        # R = np.array(R)

        self.P = P
        self.R = R

        # solve the MDP, with an extra threshold to guarantee value iteration
        # ends if gamma=1
        v, pi, Vals, R_expand = SolveMDP(
            P, R, gamma, epsilon, print_sol, prob, threshold=self.t_max * self.r_max * 3
        )  # threshold is only used when gamma=1?

        # store values and policies and matrices
        self.v = v
        self.pi = pi

        return v, pi, Vals, R_expand

    # solve_MDP() takes the trained model as well as parameters for gamma,
    # epsilon, whether the problem is a minimization or maximization one,
    # and the threshold cutoffs to not include actions that don't appear enough
    # in each state, as well as purity cutoff for next_states that do not
    # represent enough percentage of all the potential next_states,
    # and returns the the value and policy. When solving the MDP, creates an
    # artificial punishment state that is reached for state/action pairs that
    # don't meet the above cutoffs; also creates a sink node of reward 0
    # after the goal state or punishment state is reached.
    def solve_MDP_Robust(
        self,
        stratified,
        removed_actions=[],
        # alpha = 0.2, # statistical alpha threshold
        # beta = 0.6, # statistical beta threshold
        # min_action_obs = -1, # int: least number of actions that must be seen
        # min_action_purity = 0.3, # float: percentage purity above which is acceptable
        prob="min",  # str: 'max', or 'min' for maximization or minimization problem
        lever=1,
        gamma=0.9,  # discount factor
        epsilon=10 ** (-10),
        print_sol=True,
        prob_thresh=0,
    ):
        # if default value, then scale the min threshold with data size, ratio 0.008
        # min number of times a cluster/action/next_cluster transition must be observed
        # if min_action_obs == -1: #what is this used for
        #     min_action_obs = max(5, 0.008*self.df_trained.shape[0])

        # R and P get modified (with punishment node)
        # R_df and P_df stay the same

        if stratified:
            print("stratified")
            R = self.R_df_stratified.copy()
            P_df = self.P_df_stratified.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc_stratified['count']
            # P_df['purity'] = self.nc_stratified['purity']
            P_df = P_df.reset_index()

        else:
            # adding two clusters: one for sink node (reward = 0), one for punishment state
            # sink node is R[s-1], punishment state is R[s]
            R = self.R_df.copy()
            P_df = self.P_df.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc['count']
            # P_df['purity'] = self.nc['purity']
            P_df = P_df.reset_index()

        num_a = P_df["ACTION"].nunique()  # number of actions
        num_s = len(
            list(set(P_df["CLUSTER"]).union(set(P_df["NEXT_CLUSTER"])))
        )  # number of clusters, already includes the sink node
        actions = list(P_df["ACTION"].unique())
        print("num_actions: ", num_a)
        print("num clusters: ", num_s)

        # These weren't made for stochastic P_df
        # Take out rows that don't pass statistical alpha test # what's the statistical alpha test
        # P_alph = P_df.loc[(1-binom.cdf(P_df['purity']*(P_df['count']), P_df['count'],\
        #                               beta))<=alpha]

        # # Take out rows where actions or purity below threshold # what exactly is count and purity
        # P_thresh = P_alph.loc[(P_alph['count']>min_action_obs)&(P_alph['purity']>min_action_purity)]

        print("incomplete clusters and missing actions")
        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(
            P_df.groupby("CLUSTER")["ACTION"].nunique() < num_a
        )[
            0
        ]  # now it does what it's supposed to with nunique
        # stores tuples of clusters and missing action
        print(incomplete_clusters)
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(
                actions, P_df.loc[P_df["CLUSTER"] == c]["ACTION"].unique()
            )
            print("cluster: ", c, "missing actions: ", not_present, len(not_present))
            for a in not_present:
                missing_pairs.append((c, a))
        # print(missing_pairs)
        # print(len(missing_pairs))
        print("---------------------------------------------------------------")

        # printing the observed actions and count for each cluster
        # for c in range(num_s):
        #     print(P_df.loc[P_df['CLUSTER']==c]['ACTION'].unique())

        # an additional punishment node (in addition to sink node) is added for actions we don't ever observe - we don't ever want to take these actions
        # 'CLUSTER' and 'NEXT_CLUSTER' are label encoded even for stratified, so index[0] and index[2] will be numbers
        P = np.zeros((num_a, num_s + 1, num_s + 1))
        for index in P_df_1.index:
            P[index[1], index[0], index[2]] = P_df_1.loc[
                index
            ]  # action, cluster, next_cluster

        # punishment node = num_s
        # print('current R indices: ', R.index, 'new sink node: ', num_s-1, 'new punishment state: ', num_s)
        if prob == "max":
            R.loc[num_s] = -100000000
        if prob == "min":
            R.loc[num_s] = (
                100000000  # s should be the name of the punishment node and have a reward of infinity
            )

        for a in removed_actions:
            P[a, :, :] = 0
            P[a, :, -1] = 1
        # # reinsert transition for cluster/action pairs taken out by alpha test (to the punishment node)
        # excl_alph = P_df.loc[(1-binom.cdf(P_df['purity']*P_df['count'], P_df['count'],\
        #                               beta))>alpha]
        # for row in excl_alph.itertuples():
        #     c, u = row[1], row[2] #CLUSTER, ACTION
        #     P[u, c, -1] = 1 # they go directly to the punishment node

        # # reinsert transition for cluster/action pairs taken out by threshold
        # excl = P_df.loc[(P_df['count']<=min_action_obs)|(P_df['purity']<=min_action_purity)]
        # for row in excl.itertuples():
        #     c, u = row[1], row[2] #CLUSTER, ACTION
        #     P[u, c, -1] = 1

        # reinsert transition for missing cluster-action pairs (goes to punishment node)
        # print("the missing pairs transition to the punishment node")
        for pair in missing_pairs:
            c, a = pair
            P[a, c, -1] = 1
            # print(pair)
            # print(P[a,c,num_s])
        # print("-------------------------------------------------------")

        # replacing correct sink node transitions ??? idk what this is
        # nan = P_df.loc[P_df['count'].isnull()]
        # print(nan)
        # for row in nan.itertuples():
        #     c, u, t = row[1], row[2], row[3] #CLUSTER, ACTION, NEXT_CLUSTER
        #     P[u, c, t] = 1

        # punishment node to 0 reward sink (if sink was created in get_MDP):
        # if 'End' in self.df_trained['NEXT_CLUSTER'].unique():
        # print("For every action, punishment node transitions to sink node")
        for a in range(num_a):
            P[a, -1, -2] = 1  # punishment node transitions to sink node?
            # print(a, P[a,num_s,num_s-1])
        # print("-----------------------------------------------------------")

        # PERHAPS THIS SHOULD BE AFTER EVERYTHING
        # basically doing the same thing as reinsert missing cluster-action pairs below
        # quick bandaid fix for probabilities not summing to 1
        print("any cluster action pair whose transition probabilites don't sum to one")
        for a in range(P.shape[0]):
            for c in range(P.shape[1]):
                if sum(P[a, c, :]) < 0.99:
                    print(c, a, sum(P[a, c, :]))
                    P[a, c, :] = 0
                    P[a, c, -1] = (
                        1  # for actions we don't know what happens next, send them to the punishment node
                    )
        print("------------------------------------------------------------")

        # append high negative reward for incorrect / impure transitions # WDTM
        # R = []
        T_max = self.df_trained["TIME"].max()
        r_max = abs(self.df_trained["RISK"]).max()
        self.t_max = T_max
        self.r_max = r_max
        # for i in range(a):
        #     if prob == 'max':
        #         # take T-max * max(abs(reward)) * 2
        #         R.append(np.append(np.array(self.R_df),-self.t_max*self.r_max*2))
        #     else:
        #         R.append(np.append(np.array(self.R_df),self.t_max*self.r_max*2))
        # R = np.array(R)

        if stratified:
            self.P_stratified = P
            self.R_stratified = R
            v, pi, Vals = SolveMDP_Robust(
                P,
                R,
                prob_thresh,
                gamma,
                epsilon,
                print_sol,
                prob,
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            self.v_stratified = v
            self.pi_stratified = pi
        else:
            self.P = P
            # self.P_df = pd.DataFrame(P) #gives an error about 3d input
            self.R = R
            # solve the MDP, with an extra threshold to guarantee value iteration
            # ends if gamma=1
            #                        SolveMDP_Robust(P,R, lever=1, prob_thresh = 0, gamma=0.9, epsilon=10**(-10), p=True, prob='min', method='Value', threshold=float('inf')
            v, pi, Vals, R_expand = SolveMDP_Robust(
                P,
                R,
                lever,
                prob_thresh,
                gamma,
                epsilon,
                print_sol,
                prob,
                method="Value",
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            # store values and policies and matrices
            self.v = v
            self.pi = pi

        return v, pi, Vals, R_expand

    # solve_MDP() takes the trained model as well as parameters for gamma,
    # epsilon, whether the problem is a minimization or maximization one,
    # and the threshold cutoffs to not include actions that don't appear enough
    # in each state, as well as purity cutoff for next_states that do not
    # represent enough percentage of all the potential next_states,
    # and returns the the value and policy. When solving the MDP, creates an
    # artificial punishment state that is reached for state/action pairs that
    # don't meet the above cutoffs; also creates a sink node of reward 0
    # after the goal state or punishment state is reached.
    def solve_MDP_Robust_quantiles(
        self,
        stratified,
        removed_actions=[],
        prob="min",  # str: 'max', or 'min' for maximization or minimization problem
        percentile=50,
        gamma=0.9,  # discount factor
        epsilon=10 ** (-10),
        print_sol=True,
        prob_thresh=0,
    ):
        # if default value, then scale the min threshold with data size, ratio 0.008
        # min number of times a cluster/action/next_cluster transition must be observed
        # if min_action_obs == -1: #what is this used for
        #     min_action_obs = max(5, 0.008*self.df_trained.shape[0])

        # R and P get modified (with punishment node)
        # R_df and P_df stay the same

        if stratified:
            print("stratified")
            R = self.R_df_stratified.copy()
            P_df = self.P_df_stratified.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc_stratified['count']
            # P_df['purity'] = self.nc_stratified['purity']
            P_df = P_df.reset_index()

        else:
            # adding two clusters: one for sink node (reward = 0), one for punishment state
            # sink node is R[s-1], punishment state is R[s]
            R = self.R_df.copy()
            P_df = self.P_df.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc['count']
            # P_df['purity'] = self.nc['purity']
            P_df = P_df.reset_index()

        num_a = P_df["ACTION"].nunique()  # number of actions
        num_s = len(
            list(set(P_df["CLUSTER"]).union(set(P_df["NEXT_CLUSTER"])))
        )  # number of clusters, already includes the sink node
        actions = list(P_df["ACTION"].unique())
        print("num_actions: ", num_a)
        print("num clusters: ", num_s)

        # These weren't made for stochastic P_df
        # Take out rows that don't pass statistical alpha test # what's the statistical alpha test
        # P_alph = P_df.loc[(1-binom.cdf(P_df['purity']*(P_df['count']), P_df['count'],\
        #                               beta))<=alpha]

        # # Take out rows where actions or purity below threshold # what exactly is count and purity
        # P_thresh = P_alph.loc[(P_alph['count']>min_action_obs)&(P_alph['purity']>min_action_purity)]

        print("incomplete clusters and missing actions")
        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(
            P_df.groupby("CLUSTER")["ACTION"].nunique() < num_a
        )[
            0
        ]  # now it does what it's supposed to with nunique
        # stores tuples of clusters and missing action
        print(incomplete_clusters)
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(
                actions, P_df.loc[P_df["CLUSTER"] == c]["ACTION"].unique()
            )
            print("cluster: ", c, "missing actions: ", not_present, len(not_present))
            for a in not_present:
                missing_pairs.append((c, a))
        # print(missing_pairs)
        # print(len(missing_pairs))
        print("---------------------------------------------------------------")

        # printing the observed actions and count for each cluster
        # for c in range(num_s):
        #     print(P_df.loc[P_df['CLUSTER']==c]['ACTION'].unique())

        # an additional punishment node (in addition to sink node) is added for actions we don't ever observe - we don't ever want to take these actions
        # 'CLUSTER' and 'NEXT_CLUSTER' are label encoded even for stratified, so index[0] and index[2] will be numbers
        P = np.zeros((num_a, num_s + 1, num_s + 1))
        for index in P_df_1.index:
            P[index[1], index[0], index[2]] = P_df_1.loc[
                index
            ]  # action, cluster, next_cluster

        # punishment node = num_s
        # print('current R indices: ', R.index, 'new sink node: ', num_s-1, 'new punishment state: ', num_s)
        if prob == "max":
            R.loc[num_s] = -100000000
        if prob == "min":
            R.loc[num_s] = (
                100000000  # s should be the name of the punishment node and have a reward of infinity
            )

        for a in removed_actions:
            P[a, :, :] = 0
            P[a, :, -1] = 1
        # # reinsert transition for cluster/action pairs taken out by alpha test (to the punishment node)
        # excl_alph = P_df.loc[(1-binom.cdf(P_df['purity']*P_df['count'], P_df['count'],\
        #                               beta))>alpha]
        # for row in excl_alph.itertuples():
        #     c, u = row[1], row[2] #CLUSTER, ACTION
        #     P[u, c, -1] = 1 # they go directly to the punishment node

        # # reinsert transition for cluster/action pairs taken out by threshold
        # excl = P_df.loc[(P_df['count']<=min_action_obs)|(P_df['purity']<=min_action_purity)]
        # for row in excl.itertuples():
        #     c, u = row[1], row[2] #CLUSTER, ACTION
        #     P[u, c, -1] = 1

        # reinsert transition for missing cluster-action pairs (goes to punishment node)
        # print("the missing pairs transition to the punishment node")
        for pair in missing_pairs:
            c, a = pair
            P[a, c, -1] = 1
            # print(pair)
            # print(P[a,c,num_s])
        # print("-------------------------------------------------------")

        # replacing correct sink node transitions ??? idk what this is
        # nan = P_df.loc[P_df['count'].isnull()]
        # print(nan)
        # for row in nan.itertuples():
        #     c, u, t = row[1], row[2], row[3] #CLUSTER, ACTION, NEXT_CLUSTER
        #     P[u, c, t] = 1

        # punishment node to 0 reward sink (if sink was created in get_MDP):
        # if 'End' in self.df_trained['NEXT_CLUSTER'].unique():
        # print("For every action, punishment node transitions to sink node")
        for a in range(num_a):
            P[a, -1, -2] = 1  # punishment node transitions to sink node?
            # print(a, P[a,num_s,num_s-1])
        # print("-----------------------------------------------------------")

        # PERHAPS THIS SHOULD BE AFTER EVERYTHING
        # basically doing the same thing as reinsert missing cluster-action pairs below
        # quick bandaid fix for probabilities not summing to 1
        print("any cluster action pair whose transition probabilites don't sum to one")
        for a in range(P.shape[0]):
            for c in range(P.shape[1]):
                if sum(P[a, c, :]) < 0.99:
                    print(c, a, sum(P[a, c, :]))
                    P[a, c, :] = 0
                    P[a, c, -1] = (
                        1  # for actions we don't know what happens next, send them to the punishment node
                    )
        print("------------------------------------------------------------")

        # append high negative reward for incorrect / impure transitions # WDTM
        # R = []
        T_max = self.df_trained["TIME"].max()
        r_max = abs(self.df_trained["RISK"]).max()
        self.t_max = T_max
        self.r_max = r_max
        # for i in range(a):
        #     if prob == 'max':
        #         # take T-max * max(abs(reward)) * 2
        #         R.append(np.append(np.array(self.R_df),-self.t_max*self.r_max*2))
        #     else:
        #         R.append(np.append(np.array(self.R_df),self.t_max*self.r_max*2))
        # R = np.array(R)

        if stratified:
            self.P_stratified = P
            self.R_stratified = R
            v, pi, Vals = SolveMDP_Robust_quantiles(
                P,
                R,
                prob_thresh,
                gamma,
                epsilon,
                print_sol,
                prob,
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            self.v_stratified = v
            self.pi_stratified = pi
        else:
            self.P = P
            # self.P_df = pd.DataFrame(P) #gives an error about 3d input
            self.R = R
            # solve the MDP, with an extra threshold to guarantee value iteration
            # ends if gamma=1
            v, pi, Vals, R_expand = SolveMDP_Robust_quantiles(
                P,
                R,
                percentile,
                prob_thresh,
                gamma,
                epsilon,
                print_sol,
                prob,
                "Value",
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            # store values and policies and matrices
            self.v = v
            self.pi = pi

        return v, pi, Vals, R_expand

    # and the threshold cutoffs to not include actions that don't appear enough
    # in each state, as well as purity cutoff for next_states that do not
    # represent enough percentage of all the potential next_states,
    # and returns the the value and policy. When solving the MDP, creates an
    # artificial punishment state that is reached for state/action pairs that
    # don't meet the above cutoffs; also creates a sink node of reward 0
    # after the goal state or punishment state is reached.
    def solve_MDP_ICTE(
        self,
        stratified,
        removed_actions=[],
        prob="min",  # str: 'max', or 'min' for maximization or minimization problem
        percentile=50,
        gamma=0.9,  # discount factor
        epsilon=10 ** (-10),
        print_sol=True,
        prob_thresh=0,
    ):
        # if default value, then scale the min threshold with data size, ratio 0.008
        # min number of times a cluster/action/next_cluster transition must be observed
        # if min_action_obs == -1: #what is this used for
        #     min_action_obs = max(5, 0.008*self.df_trained.shape[0])

        # R and P get modified (with punishment node)
        # R_df and P_df stay the same

        if stratified:
            print("stratified")
            R = self.R_df_stratified.copy()
            P_df = self.P_df_stratified.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc_stratified['count']
            # P_df['purity'] = self.nc_stratified['purity']
            P_df = P_df.reset_index()

        else:
            # adding two clusters: one for sink node (reward = 0), one for punishment state
            # sink node is R[s-1], punishment state is R[s]
            R = self.R_df.copy()
            P_df = self.P_df.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc['count']
            # P_df['purity'] = self.nc['purity']
            P_df = P_df.reset_index()

        num_a = P_df["ACTION"].nunique()  # number of actions
        num_s = len(
            list(set(P_df["CLUSTER"]).union(set(P_df["NEXT_CLUSTER"])))
        )  # number of clusters, already includes the sink node
        actions = list(P_df["ACTION"].unique())
        if print_sol == True:
            print("num_actions: ", num_a)
            print("num clusters: ", num_s)

        # These weren't made for stochastic P_df
        # Take out rows that don't pass statistical alpha test # what's the statistical alpha test
        # P_alph = P_df.loc[(1-binom.cdf(P_df['purity']*(P_df['count']), P_df['count'],\
        #                               beta))<=alpha]

        # # Take out rows where actions or purity below threshold # what exactly is count and purity
        # P_thresh = P_alph.loc[(P_alph['count']>min_action_obs)&(P_alph['purity']>min_action_purity)]
        if print_sol == True:
            print("incomplete clusters and missing actions")
        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(
            P_df.groupby("CLUSTER")["ACTION"].nunique() < num_a
        )[
            0
        ]  # now it does what it's supposed to with nunique
        # stores tuples of clusters and missing action
        if print_sol == True:
            print(incomplete_clusters)
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(
                actions, P_df.loc[P_df["CLUSTER"] == c]["ACTION"].unique()
            )
            if print_sol == True:
                print(
                    "cluster: ", c, "missing actions: ", not_present, len(not_present)
                )
            for a in not_present:
                missing_pairs.append((c, a))
        # print(missing_pairs)
        # print(len(missing_pairs))
        print("---------------------------------------------------------------")

        # printing the observed actions and count for each cluster
        # for c in range(num_s):
        #     print(P_df.loc[P_df['CLUSTER']==c]['ACTION'].unique())

        # an additional punishment node (in addition to sink node) is added for actions we don't ever observe - we don't ever want to take these actions
        # 'CLUSTER' and 'NEXT_CLUSTER' are label encoded even for stratified, so index[0] and index[2] will be numbers
        P = np.zeros((num_a, num_s + 1, num_s + 1))
        for index in P_df_1.index:
            P[index[1], index[0], index[2]] = P_df_1.loc[
                index
            ]  # action, cluster, next_cluster

        # punishment node = num_s
        # print('current R indices: ', R.index, 'new sink node: ', num_s-1, 'new punishment state: ', num_s)
        if prob == "max":
            R.loc[num_s] = -100000000
        if prob == "min":
            R.loc[num_s] = (
                100000000  # s should be the name of the punishment node and have a reward of infinity
            )

        for a in removed_actions:
            P[a, :, :] = 0
            P[a, :, -1] = 1
        # # reinsert transition for cluster/action pairs taken out by alpha test (to the punishment node)
        # excl_alph = P_df.loc[(1-binom.cdf(P_df['purity']*P_df['count'], P_df['count'],\
        #                               beta))>alpha]
        # for row in excl_alph.itertuples():
        #     c, u = row[1], row[2] #CLUSTER, ACTION
        #     P[u, c, -1] = 1 # they go directly to the punishment node

        # # reinsert transition for cluster/action pairs taken out by threshold
        # excl = P_df.loc[(P_df['count']<=min_action_obs)|(P_df['purity']<=min_action_purity)]
        # for row in excl.itertuples():
        #     c, u = row[1], row[2] #CLUSTER, ACTION
        #     P[u, c, -1] = 1

        # reinsert transition for missing cluster-action pairs (goes to punishment node)
        # print("the missing pairs transition to the punishment node")
        for pair in missing_pairs:
            c, a = pair
            P[a, c, -1] = 1
            # print(pair)
            # print(P[a,c,num_s])
        # print("-------------------------------------------------------")

        # replacing correct sink node transitions ??? idk what this is
        # nan = P_df.loc[P_df['count'].isnull()]
        # print(nan)
        # for row in nan.itertuples():
        #     c, u, t = row[1], row[2], row[3] #CLUSTER, ACTION, NEXT_CLUSTER
        #     P[u, c, t] = 1

        # punishment node to 0 reward sink (if sink was created in get_MDP):
        # if 'End' in self.df_trained['NEXT_CLUSTER'].unique():
        # print("For every action, punishment node transitions to sink node")
        for a in range(num_a):
            P[a, -1, -2] = 1  # punishment node transitions to sink node?
            # print(a, P[a,num_s,num_s-1])
        # print("-----------------------------------------------------------")

        # PERHAPS THIS SHOULD BE AFTER EVERYTHING
        # basically doing the same thing as reinsert missing cluster-action pairs below
        # quick bandaid fix for probabilities not summing to 1
        if print_sol == True:
            print(
                "any cluster action pair whose transition probabilites don't sum to one"
            )
        for a in range(P.shape[0]):
            for c in range(P.shape[1]):
                if sum(P[a, c, :]) < 0.99:
                    if print_sol == True:
                        print(c, a, sum(P[a, c, :]))
                    P[a, c, :] = 0
                    P[a, c, -1] = (
                        1  # for actions we don't know what happens next, send them to the punishment node
                    )
        print("------------------------------------------------------------")

        # append high negative reward for incorrect / impure transitions # WDTM
        # R = []
        T_max = self.df_trained["TIME"].max()
        r_max = abs(self.df_trained["RISK"]).max()
        self.t_max = T_max
        self.r_max = r_max
        # for i in range(a):
        #     if prob == 'max':
        #         # take T-max * max(abs(reward)) * 2
        #         R.append(np.append(np.array(self.R_df),-self.t_max*self.r_max*2))
        #     else:
        #         R.append(np.append(np.array(self.R_df),self.t_max*self.r_max*2))
        # R = np.array(R)

        if stratified:
            self.P_stratified = P
            self.R_stratified = R
            v, pi, Vals = SolveMDP_ICTE(
                P,
                R,
                prob_thresh,
                gamma,
                epsilon,
                print_sol,
                prob,
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            self.v_stratified = v
            self.pi_stratified = pi
        else:
            self.P = P
            # self.P_df = pd.DataFrame(P) #gives an error about 3d input
            self.R = R
            # solve the MDP, with an extra threshold to guarantee value iteration
            # ends if gamma=1
            # SolveMDP_ICTE(P,R, percentile=50, prob_thresh = 0, gamma=0.9, epsilon=10**(-10), p=True, prob='min', method='Value', threshold=float('inf'))
            v, pi, Vals, R_expand = SolveMDP_ICTE(
                P,
                R,
                percentile,
                prob_thresh,
                gamma,
                epsilon,
                print_sol,
                prob,
                "Value",
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            # store values and policies and matrices
            self.v = v
            self.pi = pi

        return v, pi, Vals, R_expand

    def solve_MDP_Robust_expected(
        self,
        stratified,
        P_method,
        alpha=0.2,  # statistical alpha threshold
        beta=0.6,  # statistical beta threshold
        # min_action_obs = -1, # int: least number of actions that must be seen? That's not what this is??
        # min_action_purity = 0.3, # float: percentage purity above which is acceptable
        prob="min",  # str: 'max', or 'min' for maximization or minimization problem
        gamma=0.9,  # discount factor
        epsilon=10 ** (-10),  # value iteration convergence tolerance
        print_sol=True,
    ):  # print optimal value and policy
        # if default value, then scale the min threshold with data size, ratio 0.008
        # min number of times a cluster/action/next_cluster transition must be observed
        # if min_action_obs == -1: #what is this used for
        #     min_action_obs = max(5, 0.008*self.df_trained.shape[0])

        # R and P get modified (with punishment node)
        # R_df and P_df stay the same

        if stratified:
            print("stratified")
            R = self.R_df_stratified.copy()
            P_df = self.P_df_stratified.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc_stratified['count']
            # P_df['purity'] = self.nc_stratified['purity']
            P_df = P_df.reset_index()

        else:
            if P_method == 1:
                P_df = self.P_df_1.copy()
            elif P_method == 2:
                P_df = self.P_df_2.copy()
            R = self.R_df.copy()
            P_df_1 = P_df.copy()
            # P_df['count'] = self.nc['count']
            # P_df['purity'] = self.nc['purity']
            P_df = P_df.reset_index()

        num_a = P_df["ACTION"].nunique()  # number of actions
        num_s = len(
            list(set(P_df["CLUSTER"]).union(set(P_df["NEXT_CLUSTER"])))
        )  # number of clusters, already includes the sink node
        actions = list(P_df["ACTION"].unique())
        print("num_actions: ", num_a)
        print("num clusters: ", num_s)

        # These weren't made for stochastic P_df
        # Take out rows that don't pass statistical alpha test # what's the statistical alpha test
        # P_alph = P_df.loc[(1-binom.cdf(P_df['purity']*(P_df['count']), P_df['count'],\
        #                               beta))<=alpha]

        # # Take out rows where actions or purity below threshold # what exactly is count and purity
        # P_thresh = P_alph.loc[(P_alph['count']>min_action_obs)&(P_alph['purity']>min_action_purity)]

        print("incomplete clusters and missing actions")
        # Take note of rows where we have missing actions:
        incomplete_clusters = np.where(
            P_df.groupby("CLUSTER")["ACTION"].nunique() < num_a
        )[
            0
        ]  # now it does what it's supposed to with nunique
        # stores tuples of clusters and missing action
        print(incomplete_clusters)
        missing_pairs = []
        for c in incomplete_clusters:
            not_present = np.setdiff1d(
                actions, P_df.loc[P_df["CLUSTER"] == c]["ACTION"].unique()
            )
            print("cluster: ", c, "missing actions: ", not_present, len(not_present))
            for a in not_present:
                missing_pairs.append((c, a))
        # print(missing_pairs)
        # print(len(missing_pairs))
        print("---------------------------------------------------------------")

        # printing the observed actions and count for each cluster
        # for c in range(num_s):
        #     print(P_df.loc[P_df['CLUSTER']==c]['ACTION'].unique())

        # an additional punishment node (in addition to sink node) is added for actions we don't ever observe - we don't ever want to take these actions
        # 'CLUSTER' and 'NEXT_CLUSTER' are label encoded even for stratified, so index[0] and index[2] will be numbers
        P = np.zeros((num_a, num_s + 1, num_s + 1))
        for index in P_df_1.index:
            P[index[1], index[0], index[2]] = P_df_1.loc[
                index
            ]  # action, cluster, next_cluster

        # punishment node = num_s
        # print('current R indices: ', R.index, 'new sink node: ', num_s-1, 'new punishment state: ', num_s)
        if prob == "max":
            R.loc[num_s] = -100000000
        if prob == "min":
            R.loc[num_s] = (
                100000000  # s should be the name of the punishment node and have a reward of infinity
            )

        # reinsert transition for missing cluster-action pairs (goes to punishment node)
        # print("the missing pairs transition to the punishment node")
        for pair in missing_pairs:
            c, a = pair
            P[a, c, -1] = 1
            # print(pair)
            # print(P[a,c,num_s])
        # print("-------------------------------------------------------")

        # punishment node to 0 reward sink (if sink was created in get_MDP):
        # if 'End' in self.df_trained['NEXT_CLUSTER'].unique():
        # print("For every action, punishment node transitions to sink node")
        for a in range(num_a):
            P[a, -1, -2] = 1  # punishment node transitions to sink node?
            # print(a, P[a,num_s,num_s-1])
        # print("-----------------------------------------------------------")

        # PERHAPS THIS SHOULD BE AFTER EVERYTHING
        # basically doing the same thing as reinsert missing cluster-action pairs below
        # quick bandaid fix for probabilities not summing to 1
        print("any cluster action pair whose transition probabilites don't sum to one")
        for a in range(P.shape[0]):
            for c in range(P.shape[1]):
                if sum(P[a, c, :]) < 0.99:
                    print(c, a, sum(P[a, c, :]))
                    P[a, c, :] = 0
                    P[a, c, -1] = (
                        1  # for actions we don't know what happens next, send them to the punishment node
                    )
        print("------------------------------------------------------------")

        # append high negative reward for incorrect / impure transitions # WDTM
        # R = []
        T_max = self.df_trained["TIME"].max()
        r_max = abs(self.df_trained["RISK"]).max()
        self.t_max = T_max
        self.r_max = r_max
        # for i in range(a):
        #     if prob == 'max':
        #         # take T-max * max(abs(reward)) * 2
        #         R.append(np.append(np.array(self.R_df),-self.t_max*self.r_max*2))
        #     else:
        #         R.append(np.append(np.array(self.R_df),self.t_max*self.r_max*2))
        # R = np.array(R)
        if stratified:
            self.P_stratified = P
            self.R_stratified = R
            v, pi, Vals = SolveMDP_Robust_expected(
                P,
                R,
                gamma,
                epsilon,
                print_sol,
                prob,
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            self.v_stratified = v
            self.pi_stratified = pi
        else:
            self.P = P
            # self.P_df = pd.DataFrame(P) #gives an error about 3d input
            self.R = R
            # solve the MDP, with an extra threshold to guarantee value iteration
            # ends if gamma=1
            v, pi, Vals, R_expand = SolveMDP_Robust_expected(
                P,
                R,
                gamma,
                epsilon,
                print_sol,
                prob,
                threshold=self.t_max * self.r_max * 3,
            )  # threshold is only used when gamma=1?
            # store values and policies and matrices
            self.v = v
            self.pi = pi

        return v, pi, Vals, R_expand

    # opt_model_trajectory() takes a start state, a transition function,
    # indices of features to be considered, a transition function, and an int
    # for number of points to be plotted. Plots and returns the transitions
    def opt_model_trajectory(
        self,
        x,  # start state as tuple or array
        f,  # transition function of the form f(x, u) = x'
        f1=0,  # index of feature 1 to be plotted
        f2=1,  # index of feature 2 to be plotted
        n=30,
    ):  # points to be plotted
        xs, ys, all_vecs = model_trajectory(self, f, x, f1, f2, n)
        return xs, ys

    # update_predictor
    def update_predictor(self, predictor):
        self.m = predictor
        return
