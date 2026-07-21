# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 22:27:02 2020

@author: omars
"""

#################################################################
# import mdptoolbox, mdptoolbox.example
# from gurobipy import *
import numpy as np
import scipy.sparse as _sp
from numpy import random
from random import choices


#################################################################
# process P and R series into arrays (format usable by SolveMDP)
def makePandR_arrays(full_P_series, P_series, R_series, problem, verbose=False):
    """
    process P and R series into complete arrays usable by SolveMDP (inserts unseen cluster-action pairs, punishment node)

    Inputs:
    full_P_series: (series) transition probability dataframe/series from get_MDP_stochastic that includes complete list of states and actions
    P_series: (series) transition probability dataframe/series from get_MDP_stochastic to be converted into array
    R_series: (series) reward vector for each state
    problem: 'min' or 'max'
    verbose: True if printouts are wanted

    Outputs:
    P: P_series converted into an array of size NUM_ACTIONS x NUM_STATES+1 x NUM_STATES+1 (where the sizes come from full_P_series)
    R: R_series converted into an array of size NUM_STATES+1 (appends reward of punishment node)

    """

    # P should be multi-index (cluster, action, next_cluster) with 1 column (prob)
    # R index should be cluster

    # P[a, s, s']

    R = R_series.copy()
    P_df_1 = P_series.copy()  # not really a df
    P_df = P_series.reset_index()
    full_P_df = full_P_series.reset_index()

    # record parameters of transition dataframe
    num_a = full_P_df["ACTION"].nunique()
    num_s = len(
        list(set(full_P_df["CLUSTER"]).union(set(full_P_df["NEXT_CLUSTER"])))
    )  # number of clusters, already includes the sink node #P_df['CLUSTER'].nunique() #
    actions = list(P_df["ACTION"].unique())

    if verbose == True:
        print("incomplete clusters and missing actions")
    # Take note of rows where we have missing actions:
    incomplete_clusters = np.where(P_df.groupby("CLUSTER")["ACTION"].nunique() < num_a)[
        0
    ]  # now it does what it's supposed to with nunique
    # stores tuples of clusters and missing action
    if verbose == True:
        print(incomplete_clusters)
    missing_pairs = []
    for c in incomplete_clusters:
        not_present = np.setdiff1d(
            actions, P_df.loc[P_df["CLUSTER"] == c]["ACTION"].unique()
        )
        if verbose == True:
            print(
                "cluster: ",
                c,
                "missing actions: ",
                not_present,
                "num missing actions: ",
                len(not_present),
            )
        for a in not_present:
            missing_pairs.append((c, a))
    if verbose == True:
        print("---------------------------------------------------------------")

    P = np.zeros((num_a, num_s + 1, num_s + 1))
    counts = np.zeros((num_a, num_s + 1, num_s + 1))
    for index in P_df_1.index:
        # print(index)
        P[int(index[0]), int(index[1]), int(index[2])] = P_df_1.loc[
            index
        ]  # action, cluster, next_cluster
    #   try:
    #     counts[index[1],index[0],index[2]] = transition_counts_df.loc[index,"RISK"]
    #   except:
    #     counts[index[1],index[0],index[2]] = 0

    # punishment node = num_s
    # print('current R indices: ', R.index, 'new sink node: ', num_s-1, 'new punishment state: ', num_s)
    if problem == "max":
        R[int(num_s)] = -100000000
    if problem == "min":
        R[int(num_s)] = (
            100000000  # s should be the name of the punishment node and have a reward of infinity
        )

    # for a in removed_actions:
    #     P[a,:,:]=0
    #     P[a,:,-1]=1

    # reinsert transition for missing cluster-action pairs (goes to punishment node)
    # print("the missing pairs transition to the punishment node")
    for pair in missing_pairs:
        c, a = pair
        P[int(a), int(c), -1] = 1
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
    if verbose == True:
        print("any cluster action pair whose transition probabilites don't sum to one")
    for a in range(P.shape[0]):
        for c in range(P.shape[1]):
            if sum(P[a, c, :]) < 0.999:
                if verbose == True:
                    print(c, a, sum(P[a, c, :]))
                P[a, c, :] = 0
                P[a, c, -1] = (
                    1  # for actions we don't know what happens next, send them to the punishment node
                )
    if verbose == True:
        print("------------------------------------------------------------")

    return P, R, counts


#################################################################
# Tools for last function
def _randDense(states, actions, mask):
    # definition of transition matrix : square stochastic matrix
    P = np.zeros((actions, states, states))
    # definition of reward matrix (values between -1 and +1)
    R = np.zeros((actions, states, states))
    for action in range(actions):
        for state in range(states):
            # create our own random mask if there is no user supplied one
            if mask is None:
                m = np.random.random(states)
                r = np.random.random()
                m[m <= r] = 0
                m[m > r] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            # Make sure that there is atleast one transition in each state
            if m.sum() == 0:
                m[np.random.randint(0, states)] = 1
            P[action][state] = m * np.random.random(states)
            P[action][state] = P[action][state] / P[action][state].sum()
            R[action][state] = m * (
                2 * np.random.random(states) - np.ones(states, dtype=int)
            )
    return (P, R)


def _randSparse(states, actions, mask):
    # definition of transition matrix : square stochastic matrix
    P = [None] * actions
    # definition of reward matrix (values between -1 and +1)
    R = [None] * actions
    for action in range(actions):
        # it may be more efficient to implement this by constructing lists
        # of rows, columns and values then creating a coo_matrix, but this
        # works for now
        PP = _sp.dok_matrix((states, states))
        RR = _sp.dok_matrix((states, states))
        for state in range(states):
            if mask is None:
                m = np.random.random(states)
                m[m <= 2 / 3.0] = 0
                m[m > 2 / 3.0] = 1
            elif mask.shape == (actions, states, states):
                m = mask[action][state]  # mask[action, state, :]
            else:
                m = mask[state]
            n = int(m.sum())  # m[state, :]
            if n == 0:
                m[np.random.randint(0, states)] = 1
                n = 1
            # find the columns of the vector that have non-zero elements
            nz = m.nonzero()
            if len(nz) == 1:
                cols = nz[0]
            else:
                cols = nz[1]
            vals = np.random.random(n)
            vals = vals / vals.sum()
            reward = 2 * np.random.random(n) - np.ones(n)
            PP[state, cols] = vals
            RR[state, cols] = reward
        # PP.tocsr() takes the same amount of time as PP.tocoo().tocsr()
        # so constructing PP and RR as coo_matrix in the first place is
        # probably "better"
        P[action] = PP.tocsr()
        R[action] = RR.tocsr()
    return (P, R)


def rand(S, A, is_sparse=False, mask=None):
    # making sure the states and actions are more than one
    assert S > 1, "The number of states S must be greater than 1."
    assert A > 1, "The number of actions A must be greater than 1."
    if is_sparse:
        P, R = _randSparse(S, A, mask)
    else:
        P, R = _randDense(S, A, mask)
    return (P, R)


#################################################################


#################################################################
def Generate_random_MDP(
    n,  # State space size
    m,  # Nb of actions
    reduced=True,  # Reward of the form R[a,i] if true, R[a,i,j] if false
    reward_dep_action=False,  # Reward of the form R[i] if false, see reduced else
    deterministic=False,
):
    # n: number of states
    # m: number of actions
    # reduced: if the reward does not depend on the next state i.e R is of the form R(a,i) and not R(a,i,j)
    # Returns - P[a,i,j] probabilty of going to j from i when action is taken
    #         - R[a,i] reward when going to any j from i when action is taken if reduced is true OR R[a,i,j] reward when going to j from i when action is taken if reduced is false
    P, R = rand(n, m)
    if deterministic:
        for s in range(n):
            for a in range(m):
                u = np.argmax(P[a, s, :])
                for sp in range(n):
                    P[a, s, sp] = sp == u
    if reduced:
        Rp = np.zeros((m, n))
        for a in range(m):
            for i in range(n):
                Rp[a, i] = np.abs(np.mean(R[a, i, :]))
        if reward_dep_action == False:
            Rpp = np.zeros(n)
            for i in range(n):
                Rpp[i] = Rp[0, i]
            return P, Rpp
        else:
            return P, Rp
    else:
        return P, R


#################################################################


#################################################################
# Tools for last function
def expand(R, n, m):
    Rp = np.zeros((m, n, n))
    for a in range(m):
        for i in range(n):
            for j in range(n):
                if len(R.shape) == 2:
                    Rp[a, i, j] = R[a, i]
                elif len(R.shape) == 1:
                    Rp[a, i, j] = R.loc[i]
    return Rp


# def expand_d(R, n, m):
#     Rp = np.zeros((m, n, n))
#     for a in range(m):
#         for i in range(n):
#             for j in range(n):
#                 if len(R.shape) == 2:
#                     Rp[a, i, j] = R[a, i]
#                 elif len(R.shape) == 1:
#                     Rp[a, i, j] = R.loc[i]
#     return Rp


def find_interval(N, O_i, beta, n):  # for state i? Solving for roots?
    chisquare = st.chi2.ppf(
        beta, n - 1
    )  # should the first parameter be beta? yes. Unless doing bonferonni, then it should be beta/n
    denom = 2 * (N + chisquare)
    sqrterm = np.sqrt(chisquare * (chisquare + 4 * O_i * (N - O_i) / N))
    p_upr = (chisquare + 2 * O_i + sqrterm) / denom
    p_lwr = (chisquare + 2 * O_i - sqrterm) / denom
    return [p_lwr, p_upr]  # will p_lwr ever be negative?


def return_simultaneous_intervals(P_emp, observation_counts, confidence_mat):
    # observation_counts[a,s]
    # confidence_mat[a,s]
    numactions, numstates = observation_counts.shape
    intervals_mat = np.full((numactions, numstates, numstates), None, dtype=object)
    for i in range(numstates):
        for j in range(numstates):
            for k in range(numactions):
                N = observation_counts[
                    k, i
                ]  # number of observations from state-action pair i,k
                if (
                    N == 0
                ):  # the transition probabilities should stay the same as the empirically estimated
                    intervals_mat[k, i, j] = [P_emp[k, i, j], P_emp[k, i, j]]
                else:
                    O_i = observation_counts[k, i] * P_emp[k, i, j]
                    intervals_mat[k, i, j] = find_interval(
                        N, O_i, confidence_mat[k, i], numstates
                    )
    return intervals_mat


def generate_cvar_intervals(alpha_chosen, P_empirical, P_original):
    numactions = P_original.shape[0]
    numstates = P_original.shape[1]
    intervals_alpha_chosen = np.full(
        (numactions, numstates, numstates), None, dtype=object
    )
    upper_gap = np.full((numactions, numstates, numstates), None, dtype=float)
    for a in range(numactions):
        for s in range(numstates):
            for t in range(numstates):
                upperbound = (1 / alpha_chosen) * P_empirical[a, s, t]
                intervals_alpha_chosen[a, s, t] = [0, upperbound]
                upper_gap[a, s, t] = upperbound - P_original[a, s, t]

    return intervals_alpha_chosen, upper_gap


def inner_prob_obj(mu, p_lwr, p_upr, v):
    n = np.shape(v)[0]
    pos_part = np.where(mu - v < 0, 0, mu - v)
    return mu + np.sum(
        np.multiply(p_upr, pos_part)
        - np.multiply(p_lwr, pos_part)
        + np.multiply(v, p_upr)
        - np.multiply(mu, p_upr)
    )


def solve_inner_problem_binarySearch(v, p_lwr, p_upr):
    """
    The inner problem takes the form:
        max_{p} v^T p : p^T 1 = 1, p_lwr <= p <= p_upr
    Formulating the dual reveals a convex piecewise linear function to be minimized
    which has breakpoints v(0):=0, and v(1),...,v(n)
    The minimum is obtained at one of the breakpoints
    By ordering the breakpoints and conducting a binary search algorithm we can efficiently find the minimum
    """
    n = np.shape(v)[0]
    breakpoints = np.sort(np.insert(v, 0, 0))
    left, right = 0, len(breakpoints) - 1

    while left < right:
        mid = (left + right) // 2
        value_at_mid = inner_prob_obj(breakpoints[mid], p_lwr, p_upr, v)

        if value_at_mid > inner_prob_obj(breakpoints[mid + 1], p_lwr, p_upr, v):
            left = mid + 1
        else:
            right = mid

    min_breakpoint = breakpoints[left]
    min_value = inner_prob_obj(min_breakpoint, p_lwr, p_upr, v)
    return min_value


def robust_value_iteration(
    P, R, unc_intervals, gamma=0.9, objective="min", eps=1e-3, verbose=False
):
    """
    INPUTS:
    States: s=1,...,n
    Actions: a=1,...,m
    Transition Probabilities: P(a,s,s') for all s=1,...,n, s'=1,...,n, a=1,...,m,
    Reward Function: R(s) for all s=1,...,n
    gamma: discount factor

    OUTPUTS:
    opt_vals: a dictionary of the value for each state
    opt_actions: a dictionary of the optimal action for each state
    diff: gap between value vectors of consecutive iterations
    k: number of iterations
    """
    # if objective=='max':
    #     R = -R

    m, n, n = np.shape(P)
    k = 1
    vhat_kplus1 = np.copy(R)
    vhat_k = np.zeros(n)
    delta = ((1 - gamma) * eps) / (2 * gamma)

    diff = np.linalg.norm(vhat_kplus1 - vhat_k)

    while diff > delta:
        vhat_k = vhat_kplus1.copy()
        vhat_kplus1 = np.zeros(n)
        for s in range(n):
            nextstatevalues = []
            for a in range(m):

                if not np.any(P[a, s, :] != 0):
                    continue

                p_lwr_vals = [x[0] for x in unc_intervals[a, s, :]]
                p_upr_vals = [x[1] for x in unc_intervals[a, s, :]]

                sigma_hat_a = solve_inner_problem_binarySearch(
                    vhat_k, p_lwr_vals, p_upr_vals
                )

                nextstatevalues.append(R[s] + gamma * sigma_hat_a)

            if len(nextstatevalues) > 0:
                vhat_kplus1[s] = np.min(np.array(nextstatevalues))
            else:
                vhat_kplus1[s] = 0
        diff = np.linalg.norm(vhat_kplus1 - vhat_k)
        if (k % 10 == 0) & verbose:
            print(f"Iteration: {k}, gap: {diff}")

        k += 1

    # Extracting optimal policy
    opt_actions = np.zeros(n)
    opt_vals = np.zeros(n)
    Q_mat = np.zeros((n, m))
    for s in range(n):
        opt_vals[s] = vhat_kplus1[s]
        act_vals = []
        nextstatevalues = []
        for a in range(m):
            p_lwr_vals = [x[0] for x in unc_intervals[a, s, :]]
            p_upr_vals = [x[1] for x in unc_intervals[a, s, :]]
            sigma_hat_a = solve_inner_problem_binarySearch(
                vhat_kplus1, p_lwr_vals, p_upr_vals
            )
            Q_mat[s, a] = R[s] + gamma * sigma_hat_a
            nextstatevalues.append(R[s] + gamma * sigma_hat_a)
        opt_actions[s] = int(np.argmin(np.array(nextstatevalues)))

    if objective == "max":
        return list(-vhat_kplus1), list(opt_actions.astype(int)), Q_mat, diff, k
    else:
        return list(vhat_kplus1), list(opt_actions.astype(int)), Q_mat, diff, k


# Bellman operator for traditional value iteration
def Bell(V, P, R, gamma, prob="min", reduced=True):
    # inputs:
    # V: a vector storing the approximated value at each state under some policy
    # P: P[a,i,j] is the probability of transitioning from i to j when action a is taken
    # R: R[a,i,j] is the reward when going from i to j when action a is taken
    # m: number of actions
    # n: number of states
    m, n, n = P.shape
    res = np.zeros(n)
    v = V.copy()
    if prob == "min":
        for i in range(n):
            # if sum(P[0,i,:]) != 1:
            # print('state index: ', i, ', action index: 0', ', prob vec: ', P[0,i,:])
            res[i] = sum(P[0, i, :] * (R[0, i, :] + gamma * v))
            for a in range(m):
                # if sum(P[a,i,:]) != 1:
                # print('state index: ', i, ', action index: ',a, ', prob vec: ', P[a,i,:])
                res[i] = min(res[i], sum(P[a, i, :] * (R[a, i, :] + gamma * v)))
    if prob == "max":
        for i in range(n):
            # if sum(P[0,i,:]) != 1:
            # print('state index: ', i, ', action index: 0', ', prob vec: ', P[0,i,:])
            res[i] = sum(P[0, i, :] * (R[0, i, :] + gamma * v))
            for a in range(m):
                # if sum(P[a,i,:]) != 1:
                # print('state index: ', i, ', action index: ',a, ', prob vec: ', P[a,i,:])
                res[i] = max(res[i], sum(P[a, i, :] * (R[a, i, :] + gamma * v)))
    return res


# Traditional Value Iteration
def ValueIteration(
    P, R, gamma=0.9, epsilon=10 ** (-10), prob="min", threshold=float("inf")
):
    m, n, n = P.shape
    V = np.zeros(n)
    W = Bell(V, P, R, gamma, prob)
    its = 0
    while np.linalg.norm(V - W) > epsilon:
        its += 1
        V = W
        W = Bell(W, P, R, gamma, prob)
        if (
            gamma == 1 and max(abs(V)) > threshold
        ):  # threshold in case the value is actually infinity, used when gamma=1
            return V
    print("number of iterations: ", its)
    return W


# Getting Policy from value function
def GetPolicy(V, P, R, gamma, prob="min"):
    if len(R.shape) == 2:
        R = expand(R)
    m, n, l = P.shape
    Vals = [
        [sum(P[a, i, :] * (R[a, i, :] + gamma * V)) for a in range(m)] for i in range(n)
    ]
    # Vals[i][a] is the value of taking action a from state i, like q(s,a)
    if prob == "min":
        pi = [np.argmin(Vals[i]) for i in range(n)]
    if prob == "max":
        pi = [np.argmax(Vals[i]) for i in range(n)]
    return np.array(pi), np.array(Vals)


# Bellman operator for quantile value iteration (percentile  = quantile*100, prob=objective)
def Bell_Robust_quantiles(
    V, P, R, gamma, prob_thresh, percentile=50, prob="min", reduced=True
):
    # inputs:
    # V: a vector storing the approximated value at each state under some policy
    # P: P[a,i,j] is the probability of transitioning from i to j when action a is taken
    # R: R[a,i,j] is the reward when going from i to j when action a is taken
    # percentile: 100*quantile user wishes to optimize
    # m: number of actions
    # n: number of states
    # prob_thresh: minimum transition probability to be considered as a "possible" next state. If 0, considers all states with positive probability
    m, n, n = P.shape
    percentile_state_mat = np.zeros((n, m))
    res = np.zeros(n)
    v = V.copy()
    if prob == "min":
        for i in range(n):
            # doing everything for action 0 first
            next_states = np.where(P[0, i, :] > prob_thresh)[
                0
            ]  # next states with probability > prob_thresh of occuring from current state i after taking action 0
            next_state_probs = np.array(
                P[0, i, next_states]
            )  # the corresponding probabilities of the possible next states
            next_state_vals = np.array(
                v[next_states]
            )  # the values corresponding to each possible next state
            next_state_inds_sorted = np.argsort(
                next_state_vals
            )  # returns the indices in the order that would sort next_states # shouldn't it be sorted by values tho???
            percentile_values_vec = np.zeros(100) + max(
                next_states
            )  # next state value for each percentile from 1 to 100
            percentile_states_vec = (
                np.zeros(100)
                + next_states[np.where(next_state_vals == max(next_state_vals))[0][0]]
            )  # next state that attains each percentile value
            perc_idx = 0
            # fils in the percentile vectors starting from perc_idx
            for sub_ind in next_state_inds_sorted:
                p = next_state_probs[sub_ind]
                val = next_state_vals[sub_ind]
                orig_ind = next_states[sub_ind]  # the actual state we're working with
                num_percentiles = round(100 * p)
                percentile_values_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = val  # the next state value for the corresponding percentiles
                percentile_states_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = orig_ind  # the state that attains the corresponding percentiles
                perc_idx = min(perc_idx + num_percentiles, 100)
            # print("percentile states vec: ", percentile_states_vec)
            # print("percentile values vec: ", percentile_values_vec)

            percentile_state = int(
                percentile_states_vec[round(percentile)]
            )  # the state corresponding to the user specified percentile
            percentile_val = percentile_values_vec[
                round(percentile)
            ]  # the value of the above state
            # print("percentile: ", percentile)
            # print("percentile state: ", percentile_state)
            # print("percentile state value: ", v[percentile_state])
            # print("percentile value: ", percentile_val)
            # print("------------------------------------------------------------------------")

            if v[percentile_state] != percentile_val:
                print(v[percentile_state] - percentile_val)

            percentile_state_mat[i, 0] = percentile_state
            res[i] = R[0, i, percentile_state] + gamma * v[percentile_state]

            # then looping through the rest of the actions
            for a in range(m):
                next_states = np.where(P[a, i, :] > prob_thresh)[
                    0
                ]  # positive probability from state i under action 0
                next_state_probs = np.array(P[a, i, next_states])
                next_state_vals = np.array(v[next_states])
                next_state_inds_sorted = np.argsort(next_state_vals)
                percentile_values_vec = np.zeros(100) + max(next_states)
                percentile_states_vec = (
                    np.zeros(100)
                    + next_states[
                        np.where(next_state_vals == max(next_state_vals))[0][0]
                    ]
                )
                perc_idx = 0
                for sub_ind in next_state_inds_sorted:
                    p = next_state_probs[sub_ind]
                    val = next_state_vals[sub_ind]
                    orig_ind = next_states[sub_ind]
                    num_percentiles = round(100 * p)
                    percentile_values_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = val
                    percentile_states_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = orig_ind
                    perc_idx = min(perc_idx + num_percentiles, 100)

                # print("percentile states vec: ", percentile_states_vec)
                # print("percentile values vec: ", percentile_values_vec)

                percentile_state = int(percentile_states_vec[round(percentile)])
                percentile_val = percentile_values_vec[round(percentile)]
                # print("percentile: ", percentile)
                # print("percentile state: ", percentile_state)
                # print("percentile state value: ", v[percentile_state])
                # print("percentile value: ", percentile_val)
                # print("------------------------------------------------------------------------")
                if v[percentile_state] != percentile_val:
                    print(v[percentile_state] - percentile_val)

                percentile_state_mat[i, a] = percentile_state
                res_ia = R[a, i, percentile_state] + gamma * v[percentile_state]

                res[i] = min(
                    res[i], res_ia
                )  # keeping the action that gives the lowest value of res[i]

    if prob == "max":
        for i in range(n):
            next_states = np.where(P[0, i, :] > prob_thresh)[0]
            next_state_probs = np.array(P[0, i, next_states])
            next_state_vals = np.array(v[next_states])
            next_state_inds_sorted = np.argsort(next_state_vals)[
                ::-1
            ]  # sort in reverse order
            percentile_values_vec = np.zeros(100) + max(next_state_vals)
            percentile_states_vec = (
                np.zeros(100)
                + next_states[np.where(next_state_vals == max(next_state_vals))[0][0]]
            )
            perc_idx = 0
            for sub_ind in next_state_inds_sorted:
                p = next_state_probs[sub_ind]
                val = next_state_vals[sub_ind]
                orig_ind = next_states[sub_ind]
                num_percentiles = round(100 * p)
                percentile_values_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = val
                percentile_states_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = orig_ind
                perc_idx = min(perc_idx + num_percentiles, 100)
            # print("percentile states vec: ", percentile_states_vec)
            # print("percentile values vec: ", percentile_values_vec)

            percentile_state = int(percentile_states_vec[round(percentile)])
            percentile_val = percentile_values_vec[round(percentile)]
            # print("percentile: ", percentile)
            print("percentile state: ", percentile_state)
            print("percentile state value: ", v[percentile_state])
            print("percentile value: ", percentile_val)
            print(
                "------------------------------------------------------------------------"
            )

            percentile_state_mat[i, 0] = percentile_state
            res[i] = R[0, i, percentile_state] + gamma * v[percentile_state]

            for a in range(m):
                next_states = np.where(P[a, i, :] > prob_thresh)[
                    0
                ]  # positive probability from state i under action 0
                next_state_probs = np.array(P[a, i, next_states])
                next_state_vals = np.array(v[next_states])
                next_state_inds_sorted = np.argsort(next_state_vals)[::-1]
                percentile_values_vec = np.zeros(100) + max(next_states)
                percentile_states_vec = (
                    np.zeros(100)
                    + next_states[
                        np.where(next_state_vals == max(next_state_vals))[0][0]
                    ]
                )
                perc_idx = 0
                for sub_ind in next_state_inds_sorted:
                    p = next_state_probs[sub_ind]
                    val = next_state_vals[sub_ind]
                    orig_ind = next_states[sub_ind]
                    num_percentiles = round(100 * p)
                    percentile_values_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = val
                    percentile_states_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = orig_ind
                    perc_idx = min(perc_idx + num_percentiles, 100)

                # print("percentile states vec: ", percentile_states_vec)
                # print("percentile values vec: ", percentile_values_vec)

                percentile_state = int(percentile_states_vec[round(percentile)])
                percentile_val = percentile_values_vec[round(percentile)]
                # print("percentile: ", percentile)
                print("percentile state: ", percentile_state)
                print("percentile state value: ", v[percentile_state])
                print("percentile value: ", percentile_val)
                print(
                    "------------------------------------------------------------------------"
                )

                percentile_state_mat[i, a] = percentile_state
                res_ia = R[a, i, percentile_state] + gamma * v[percentile_state]

                res[i] = max(
                    res[i], res_ia
                )  # keeping the action that gives the lowest value of res[i]

    return res, percentile_state_mat


# Quantile Value Iteration
def ValueIteration_Robust_quantiles(
    P,
    R,
    gamma=0.9,
    epsilon=10 ** (-10),
    prob="min",
    percentile=50,
    threshold=float("inf"),
    prob_thresh=0,
):
    m, n, n = P.shape
    V = np.zeros(n)
    # Bell_Robust_quantiles(V, P, R, gamma, prob_thresh, percentile=50, prob='min', reduced=True)
    W, Ind = Bell_Robust_quantiles(V, P, R, gamma, prob_thresh, percentile, prob)
    its = 0
    while np.linalg.norm(V - W) > epsilon:
        its += 1
        V = W
        #  Bell_Robust_quantiles(V, P, R, gamma, prob_thresh, percentile=50, prob='min', reduced=True)
        W, Ind = Bell_Robust_quantiles(W, P, R, gamma, prob_thresh, percentile, prob)
        if (
            gamma == 1 and max(abs(V)) > threshold
        ):  # threshold in case the value is actually infinity, used when gamma=1
            return V
    print("number of iterations: ", its)
    return W, Ind


# Getting Policy from value function
def GetPolicy_Robust_quantiles(
    V, P, R, Ind, gamma, prob="min", percentile=50, prob_thresh=0
):
    if len(R.shape) == 2:
        R = expand(R)
    m, n, l = P.shape
    V = np.array(V)
    if prob == "min":
        Vals = [
            [R[a, i, int(Ind[i, a])] + gamma * V[int(Ind[i, a])] for a in range(m)]
            for i in range(n)
        ]
        # Vals[i][a] is the value of taking action a from state i, like q(s,a)
        # Vals = [[max(np.append(R[a, i, np.where(P[a,i,:]>prob_thresh)[0]] + gamma*V[np.where(P[a,i,:]>prob_thresh)[0]],-np.inf)) for a in range(m)] for i in range(n)]
        # Vals = [np.nan_to_num(Vals[i], nan = 0.0, posinf = np.inf, neginf = np.inf) for i in range(n)] #replacing value with inf since we dk what happens when you take that action (so never choose that action)
        pi = [np.argmin(Vals[i]) for i in range(n)]
    if prob == "max":
        Vals = [
            [R[a, i, int(Ind[i, a])] + gamma * V[int(Ind[i, a])] for a in range(m)]
            for i in range(n)
        ]
        # Vals = [[min(np.append(R[a, i, np.where(P[a,i,:]>prob_thresh)[0]] + gamma*V[np.where(P[a,i,:]>prob_thresh)[0]],np.inf)) for a in range(m)] for i in range(n)]
        # Vals = [np.nan_to_num(Vals[i], nan = 0.0, posinf = -np.inf, neginf = np.inf) for i in range(n)] #replacing with -inf since we dk what happens when you take that action (so never choose that action)
        pi = [np.argmax(Vals[i]) for i in range(n)]
    return pi, Vals


def standardize_values(values):
    total = sum(values)
    standardized_values = [(x / total) for x in values]
    return standardized_values


# Bellman operator for ICTE value iteration (percentile  = quantile*100, prob=objective)
def Bell_ICTE(V, P, R, gamma, prob_thresh=0, percentile=50, prob="min", reduced=True):
    # inputs:
    # V: a vector storing the approximated value at each state under some policy
    # P: P[a,i,j] is the probability of transitioning from i to j when action a is taken
    # R: R[a,i,j] is the reward when going from i to j when action a is taken
    # prob_thresh: the minimum probability to be considered a "possible" next state
    # percentile: the CTE will be taken over the WORST percentile % of outcomes (percentile = 100*quantile)

    # outputs:
    # res: new estimate for V
    # percentile_state_mat: nxm matrix where entry [s,a] is the next state that attains the specified percentile value for state s and action a

    # percentile = percentile-1
    quantile = percentile / 100
    m, n, n = P.shape  # m: number of actions, n: number of states
    Vals_mat = np.zeros((n, m))  # the current value of each state and action
    res = np.zeros(n)
    v = V.copy()
    if prob == "min":
        for i in range(n):
            # doing everything for action 0 first
            next_states = np.where(P[0, i, :] > prob_thresh)[
                0
            ]  # next states with probability > prob_thresh of occuring from current state i after taking action 0
            next_state_probs = np.array(
                P[0, i, next_states]
            )  # the corresponding probabilities of the possible next states
            next_state_vals = np.array(
                v[next_states]
            )  # the values corresponding to each possible next state
            next_state_inds_sorted = np.argsort(
                next_state_vals
            )  # returns the indices in the order that would sort next_state values in ASCENDING order
            percentile_values_vec = np.zeros(100) + max(
                next_states
            )  # next state value for each percentile from 1 to 100
            percentile_states_vec = (
                np.zeros(100)
                + next_states[np.where(next_state_vals == max(next_state_vals))[0][0]]
            )  # next state that attains each percentile value
            percentile_probs_vec = np.zeros(100)

            decreasing_inds_sorted = np.argsort(next_state_vals)[::-1]
            sorted_states = next_states[decreasing_inds_sorted]
            # print(sorted_states)
            sorted_vals = next_state_vals[decreasing_inds_sorted]
            # print(sorted_vals)
            sorted_probs = next_state_probs[decreasing_inds_sorted]
            # print(sorted_probs)

            tail_vals = []
            tail_probs = []
            tail_states = []
            cum_tail_prob = 0
            ind = 0
            while cum_tail_prob < quantile and ind < len(sorted_probs):
                if cum_tail_prob + sorted_probs[ind] <= quantile:
                    tail_vals.append(sorted_vals[ind])
                    tail_states.append(sorted_states[ind])
                    tail_probs.append(sorted_probs[ind])
                    cum_tail_prob += sorted_probs[ind]
                else:
                    tail_vals.append(sorted_vals[ind])
                    tail_states.append(sorted_states[ind])
                    tail_probs.append(quantile - cum_tail_prob)
                    cum_tail_prob += quantile - cum_tail_prob
                ind += 1

            # print(tail_vals)
            # print(tail_probs)
            # print(tail_states)
            # print(cum_tail_prob)

            # print("cumulative tail prob: ", cum_tail_prob)
            # print("sum of tail probs: ", sum(tail_probs))
            # print("percentile: ", percentile)

            normalized_tail_probs = np.array(tail_probs) / quantile
            # print(normalized_tail_probs)
            # print(sum(normalized_tail_probs))
            CTE = sum(tail_vals[:] * normalized_tail_probs[:])
            # print(CTE)

            perc_idx = 0
            # fils in the percentile vectors starting from perc_idx
            for sub_ind in next_state_inds_sorted:
                p = next_state_probs[sub_ind]
                val = next_state_vals[sub_ind]
                orig_ind = next_states[sub_ind]  # the actual state we're working with
                num_percentiles = round(100 * p)
                percentile_values_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = val  # the next state value for the corresponding percentiles
                percentile_states_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = int(
                    orig_ind
                )  # the state that attains the corresponding percentiles
                percentile_probs_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = p
                perc_idx = min(perc_idx + num_percentiles, 100)

            CTE2 = sum(percentile_values_vec[100 - percentile : 100]) / percentile
            # print(CTE2)

            if CTE != CTE2:
                print("difference in CTEs: ", abs(CTE - CTE2))

            percentile_state = int(
                percentile_states_vec[round(100 - percentile)]
            )  # the state corresponding to the user specified percentile
            # percentile_val = percentile_values_vec[round(100-percentile)] # the value of the above state

            res[i] = R[0, i, percentile_state] + gamma * CTE
            Vals_mat[i, 0] = R[0, i, percentile_state] + gamma * CTE

            # tail_states = np.unique(np.array(percentile_states_vec[round(100-percentile):100])).astype(int)
            # tail_vals = np.array(v[tail_states])
            # tail_probs = np.array(P[0,i,tail_states])

            # if v[percentile_state] != percentile_val:
            #     print("not aligned percentile value: ", v[percentile_state]-percentile_val)

            # percentile_state_mat[i,0] = percentile_state

            # tail_vals = next_state_vals[np.where(next_state_vals >= percentile_val)[0]]

            # tail_probs = next_state_probs[np.where(next_state_vals >= percentile_val)[0]]

            # first checking if the tail probs sum to what they're supposed to
            # if 100*sum(tail_probs) != percentile:
            #     print("actual percentile: ", 100*sum(tail_probs), ", desired percentile: ", percentile)

            # normalized_tail_probs = np.array(standardize_values(tail_probs))

            # if sum(normalized_tail_probs)!=1:
            #     print("sum not 1: ", sum(normalized_tail_probs))

            # checking if the normalization function works as desired
            # if (tail_probs/percentile != normalized_tail_probs).any():
            #     print("divided by percentile: ", tail_probs/percentile, ", sum: ", sum(tail_probs/percentile))
            #     print("normalized by sum: ", normalized_tail_probs, ", sum: ", sum(normalized_tail_probs))
            # print("tail values vector: ", tail_vals)
            # print("tail probabilities vector: ", tail_probs)
            # print("normalized tail probabilities: ", normalized_tail_probs)

            # then looping through the rest of the actions
            for a in range(m):
                # doing everything for action 0 first
                next_states = np.where(P[a, i, :] > prob_thresh)[
                    0
                ]  # next states with probability > prob_thresh of occuring from current state i after taking action 0
                next_state_probs = np.array(
                    P[a, i, next_states]
                )  # the corresponding probabilities of the possible next states
                next_state_vals = np.array(
                    v[next_states]
                )  # the values corresponding to each possible next state
                next_state_inds_sorted = np.argsort(
                    next_state_vals
                )  # returns the indices in the order that would sort next_states # shouldn't it be sorted by values tho???
                percentile_values_vec = np.zeros(100) + max(
                    next_states
                )  # next state value for each percentile from 1 to 100
                percentile_states_vec = (
                    np.zeros(100)
                    + next_states[
                        np.where(next_state_vals == max(next_state_vals))[0][0]
                    ]
                )  # next state that attains each percentile value
                percentile_probs_vec = np.zeros(100)

                decreasing_inds_sorted = np.argsort(next_state_vals)[::-1]
                sorted_states = next_states[decreasing_inds_sorted]
                # print(sorted_states)
                sorted_vals = next_state_vals[decreasing_inds_sorted]
                # print(sorted_vals)
                sorted_probs = next_state_probs[decreasing_inds_sorted]
                # print(sorted_probs)

                tail_vals = []
                tail_probs = []
                tail_states = []
                cum_tail_prob = 0
                ind = 0
                while cum_tail_prob < quantile and ind < len(sorted_probs):
                    if cum_tail_prob + sorted_probs[ind] <= quantile:
                        tail_vals.append(sorted_vals[ind])
                        tail_states.append(sorted_states[ind])
                        tail_probs.append(sorted_probs[ind])
                        cum_tail_prob += sorted_probs[ind]
                    else:
                        tail_vals.append(sorted_vals[ind])
                        tail_states.append(sorted_states[ind])
                        tail_probs.append(quantile - cum_tail_prob)
                        cum_tail_prob += quantile - cum_tail_prob
                    ind += 1

                # print(tail_vals)
                # print(tail_probs)
                # print(tail_states)
                # print(cum_tail_prob)

                # print("cumulative tail prob: ", cum_tail_prob)
                # print("sum of tail probs: ", sum(tail_probs))
                # print("percentile: ", percentile)

                normalized_tail_probs = np.array(tail_probs) / quantile
                # print(normalized_tail_probs)
                # print(sum(normalized_tail_probs))
                CTE = sum(tail_vals[:] * normalized_tail_probs[:])
                # print(CTE)

                perc_idx = 0
                # fils in the percentile vectors starting from perc_idx
                for sub_ind in next_state_inds_sorted:
                    p = next_state_probs[sub_ind]
                    val = next_state_vals[sub_ind]
                    orig_ind = next_states[
                        sub_ind
                    ]  # the actual state we're working with
                    num_percentiles = round(100 * p)
                    percentile_values_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = val  # the next state value for the corresponding percentiles
                    percentile_states_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = int(
                        orig_ind
                    )  # the state that attains the corresponding percentiles
                    percentile_probs_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = p
                    perc_idx = min(perc_idx + num_percentiles, 100)

                CTE2 = sum(percentile_values_vec[100 - percentile : 100]) / percentile
                # print(CTE2)

                if CTE != CTE2:
                    print("difference in CTEs: ", abs(CTE - CTE2))

                percentile_state = int(
                    percentile_states_vec[round(100 - percentile)]
                )  # the state corresponding to the user specified percentile
                # percentile_val = percentile_values_vec[round(100-percentile)] # the value of the above state

                res_ia = R[a, i, percentile_state] + gamma * CTE
                Vals_mat[i, a] = R[a, i, percentile_state] + gamma * CTE

                res[i] = min(
                    res[i], res_ia
                )  # keeping the action that gives the lowest value of res[i]

    if prob == "max":
        for i in range(n):
            # doing everything for action 0 first
            next_states = np.where(P[0, i, :] > prob_thresh)[
                0
            ]  # next states with probability > prob_thresh of occuring from current state i after taking action 0
            next_state_probs = np.array(
                P[0, i, next_states]
            )  # the corresponding probabilities of the possible next states
            next_state_vals = np.array(
                v[next_states]
            )  # the values corresponding to each possible next state
            next_state_inds_sorted = np.argsort(
                next_state_vals
            )  # returns the indices in the order that would sort next_state values in ASCENDING order
            percentile_values_vec = np.zeros(100) + max(
                next_states
            )  # next state value for each percentile from 1 to 100
            percentile_states_vec = (
                np.zeros(100)
                + next_states[np.where(next_state_vals == max(next_state_vals))[0][0]]
            )  # next state that attains each percentile value
            percentile_probs_vec = np.zeros(100)

            sorted_states = next_states[
                next_state_inds_sorted
            ]  # sorted in increasing order of vals
            # print(sorted_states)
            sorted_vals = next_state_vals[
                next_state_inds_sorted
            ]  # sorted in increasing order
            # print(sorted_vals)
            sorted_probs = next_state_probs[next_state_inds_sorted]
            # print(sorted_probs)

            tail_vals = []
            tail_probs = []
            tail_states = []
            cum_tail_prob = 0
            ind = 0
            while cum_tail_prob < quantile:
                if cum_tail_prob + sorted_probs[ind] <= quantile:
                    tail_vals.append(sorted_vals[ind])
                    tail_states.append(sorted_states[ind])
                    tail_probs.append(sorted_probs[ind])
                    cum_tail_prob += sorted_probs[ind]
                else:
                    tail_vals.append(sorted_vals[ind])
                    tail_states.append(sorted_states[ind])
                    tail_probs.append(quantile - cum_tail_prob)
                    cum_tail_prob += quantile - cum_tail_prob
                ind += 1

            # print(tail_vals)
            # print(tail_probs)
            # print(tail_states)
            # print(cum_tail_prob)

            # print("cumulative tail prob: ", cum_tail_prob)
            # print("sum of tail probs: ", sum(tail_probs))
            # print("percentile: ", percentile)

            normalized_tail_probs = np.array(tail_probs) / quantile
            # print(normalized_tail_probs)
            # print(sum(normalized_tail_probs))
            CTE = sum(tail_vals[:] * normalized_tail_probs[:])
            # print(CTE)

            perc_idx = 0
            # fils in the percentile vectors starting from perc_idx
            for sub_ind in next_state_inds_sorted:
                p = next_state_probs[sub_ind]
                val = next_state_vals[sub_ind]
                orig_ind = next_states[sub_ind]  # the actual state we're working with
                num_percentiles = round(100 * p)
                percentile_values_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = val  # the next state value for the corresponding percentiles
                percentile_states_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = int(
                    orig_ind
                )  # the state that attains the corresponding percentiles
                percentile_probs_vec[
                    perc_idx : min(perc_idx + num_percentiles, 100)
                ] = p
                perc_idx = min(perc_idx + num_percentiles, 100)

            CTE2 = sum(percentile_values_vec[100 - percentile : 100]) / percentile
            # print(CTE2)

            if CTE != CTE2:
                print("difference in CTEs: ", abs(CTE - CTE2))

            percentile_state = int(
                percentile_states_vec[round(100 - percentile)]
            )  # the state corresponding to the user specified percentile
            # percentile_val = percentile_values_vec[round(100-percentile)] # the value of the above state

            res[i] = R[0, i, percentile_state] + gamma * CTE
            Vals_mat[i, 0] = R[0, i, percentile_state] + gamma * CTE

            # then looping through the rest of the actions
            for a in range(m):
                # doing everything for action 0 first
                next_states = np.where(P[a, i, :] > prob_thresh)[
                    0
                ]  # next states with probability > prob_thresh of occuring from current state i after taking action 0
                next_state_probs = np.array(
                    P[a, i, next_states]
                )  # the corresponding probabilities of the possible next states
                next_state_vals = np.array(
                    v[next_states]
                )  # the values corresponding to each possible next state
                next_state_inds_sorted = np.argsort(
                    next_state_vals
                )  # returns the indices in the order that would sort next_states_vals in ASCENDING order
                percentile_values_vec = np.zeros(100) + max(
                    next_states
                )  # next state value for each percentile from 1 to 100
                percentile_states_vec = (
                    np.zeros(100)
                    + next_states[
                        np.where(next_state_vals == max(next_state_vals))[0][0]
                    ]
                )  # next state that attains each percentile value
                percentile_probs_vec = np.zeros(100)

                sorted_states = next_states[
                    next_state_inds_sorted
                ]  # sorted in increasing order of vals
                # print(sorted_states)
                sorted_vals = next_state_vals[
                    next_state_inds_sorted
                ]  # sorted in increasing order
                # print(sorted_vals)
                sorted_probs = next_state_probs[next_state_inds_sorted]
                # print(sorted_probs)

                tail_vals = []
                tail_probs = []
                tail_states = []
                cum_tail_prob = 0
                ind = 0
                while cum_tail_prob < quantile:
                    if cum_tail_prob + sorted_probs[ind] <= quantile:
                        tail_vals.append(sorted_vals[ind])
                        tail_states.append(sorted_states[ind])
                        tail_probs.append(sorted_probs[ind])
                        cum_tail_prob += sorted_probs[ind]
                    else:
                        tail_vals.append(sorted_vals[ind])
                        tail_states.append(sorted_states[ind])
                        tail_probs.append(quantile - cum_tail_prob)
                        cum_tail_prob += quantile - cum_tail_prob
                    ind += 1

                # print(tail_vals)
                # print(tail_probs)
                # print(tail_states)
                # print(cum_tail_prob)

                # print("cumulative tail prob: ", cum_tail_prob)
                # print("sum of tail probs: ", sum(tail_probs))
                # print("percentile: ", percentile)

                normalized_tail_probs = np.array(tail_probs) / quantile
                # print(normalized_tail_probs)
                # print(sum(normalized_tail_probs))
                CTE = sum(tail_vals[:] * normalized_tail_probs[:])
                # print(CTE)

                perc_idx = 0
                # fils in the percentile vectors starting from perc_idx
                for sub_ind in next_state_inds_sorted:
                    p = next_state_probs[sub_ind]
                    val = next_state_vals[sub_ind]
                    orig_ind = next_states[
                        sub_ind
                    ]  # the actual state we're working with
                    num_percentiles = round(100 * p)
                    percentile_values_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = val  # the next state value for the corresponding percentiles
                    percentile_states_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = int(
                        orig_ind
                    )  # the state that attains the corresponding percentiles
                    percentile_probs_vec[
                        perc_idx : min(perc_idx + num_percentiles, 100)
                    ] = p
                    perc_idx = min(perc_idx + num_percentiles, 100)

                CTE2 = sum(percentile_values_vec[100 - percentile : 100]) / percentile
                # print(CTE2)

                if CTE != CTE2:
                    print("difference in CTEs: ", abs(CTE - CTE2))

                percentile_state = int(
                    percentile_states_vec[round(100 - percentile)]
                )  # the state corresponding to the user specified percentile
                # percentile_val = percentile_values_vec[round(100-percentile)] # the value of the above state

                res_ia = R[a, i, percentile_state] + gamma * CTE
                Vals_mat[i, a] = R[a, i, percentile_state] + gamma * CTE

                res[i] = max(
                    res[i], res_ia
                )  # keeping the action that gives the highest value of res[i]

    return res, Vals_mat


# ICTE Value Iteration
def ValueIteration_ICTE(
    P,
    R,
    gamma=0.9,
    epsilon=10 ** (-10),
    prob="min",
    percentile=50,
    threshold=float("inf"),
    prob_thresh=0,
):
    m, n, n = P.shape
    V = np.zeros(n)

    W, Ind = Bell_ICTE(V, P, R, gamma, prob_thresh, percentile, prob)
    its = 0
    while np.linalg.norm(V - W) > epsilon:
        its += 1
        # print("iteration number: ", its)
        V = W
        #  Bell_ICTE(V, P, R, gamma, prob_thresh = 0, percentile=50, prob='min', reduced=True)
        # print("vals before bellman: ", W)
        W, Ind = Bell_ICTE(W, P, R, gamma, prob_thresh, percentile, prob)
        # print("vals after bellman: ", W)
        if (
            gamma == 1 and max(abs(V)) > threshold
        ):  # threshold in case the value is actually infinity, used when gamma=1
            return V
        # print("norm diff: ", np.linalg.norm(V-W))
    print("number of iterations: ", its)
    return W, Ind


# Getting Policy from value function
def GetPolicy_ICTE(V, P, R, Vals_mat, gamma, prob="min", percentile=50, prob_thresh=0):
    if len(R.shape) == 2:
        R = expand(R)
    m, n, l = P.shape
    V = np.array(V)
    if prob == "min":
        Vals = [[Vals_mat[i, a] for a in range(m)] for i in range(n)]
        # Vals[i][a] is the value of taking action a from state i, like q(s,a)
        # Vals = [[max(np.append(R[a, i, np.where(P[a,i,:]>prob_thresh)[0]] + gamma*V[np.where(P[a,i,:]>prob_thresh)[0]],-np.inf)) for a in range(m)] for i in range(n)]
        # Vals = [np.nan_to_num(Vals[i], nan = 0.0, posinf = np.inf, neginf = np.inf) for i in range(n)] #replacing value with inf since we dk what happens when you take that action (so never choose that action)
        pi = [np.argmin(Vals[i]) for i in range(n)]
    if prob == "max":
        Vals = [[Vals_mat[i, a] for a in range(m)] for i in range(n)]
        # Vals = [[min(np.append(R[a, i, np.where(P[a,i,:]>prob_thresh)[0]] + gamma*V[np.where(P[a,i,:]>prob_thresh)[0]],np.inf)) for a in range(m)] for i in range(n)]
        # Vals = [np.nan_to_num(Vals[i], nan = 0.0, posinf = -np.inf, neginf = np.inf) for i in range(n)] #replacing with -inf since we dk what happens when you take that action (so never choose that action)
        pi = [np.argmax(Vals[i]) for i in range(n)]
    return pi, Vals


# Bellman operator for weighted extreme values value iteration (alpha = lever, prob = objective)
def Bell_Robust(V, P, R, gamma, prob_thresh, lever=1, prob="min", reduced=True):
    # inputs:
    # V: a vector storing the approximated value at each state under some policy
    # P: P[a,i,j] is the probability of transitioning from i to j when action a is taken
    # R: R[a,i,j] is the reward when going from i to j when action a is taken
    # lever: weight between 0 and 1 to give to the maximum next state value
    # m: number of actions
    # n: number of states
    # prob_thresh: minimum transition probability to be considered as a "possible" next state. If 0, considers all states with positive probability
    m, n, n = P.shape
    res = np.zeros(n)
    v = V.copy()
    if prob == "min":
        for i in range(n):
            # initiating with action 0
            next_state_inds = np.where(P[0, i, :] > prob_thresh)[
                0
            ]  # positive probability from state i under action 0

            res[i] = lever * max(
                np.array(R[0, i, next_state_inds])
                + gamma * np.array(v[next_state_inds])
            ) + (1 - lever) * min(
                np.array(R[0, i, next_state_inds])
                + gamma * np.array(v[next_state_inds])
            )

            for a in range(m):
                next_state_inds = np.where(P[a, i, :] > prob_thresh)[0]

                res_ia = lever * max(
                    np.array(R[a, i, next_state_inds])
                    + gamma * np.array(v[next_state_inds])
                ) + (1 - lever) * min(
                    np.array(R[a, i, next_state_inds])
                    + gamma * np.array(v[next_state_inds])
                )

                res[i] = min(
                    res[i], res_ia
                )  # keeping the action that gives the lowest (best) value of res[i]

    if prob == "max":
        for i in range(n):
            # initiating with action 0
            next_state_inds = np.where(P[0, i, :] > prob_thresh)[0]

            res[i] = lever * min(
                np.array(R[0, i, next_state_inds])
                + gamma * np.array(v[next_state_inds])
            ) + (1 - lever) * max(
                np.array(R[0, i, next_state_inds])
                + gamma * np.array(v[next_state_inds])
            )

            for a in range(m):
                next_state_inds = np.where(P[a, i, :] > prob_thresh)[0]

                res_ia = lever * min(
                    np.array(R[a, i, next_state_inds])
                    + gamma * np.array(v[next_state_inds])
                ) + (1 - lever) * max(
                    np.array(R[a, i, next_state_inds])
                    + gamma * np.array(v[next_state_inds])
                )

                res[i] = max(
                    res[i], res_ia
                )  # keeping the action that gives the highest (best) value of res[i]
    return res


# Weighted Extreme Values Value Iteration
def ValueIteration_Robust(
    P,
    R,
    gamma=0.9,
    epsilon=10 ** (-10),
    prob="min",
    lever=1,
    threshold=float("inf"),
    prob_thresh=0,
):
    m, n, n = P.shape
    V = np.zeros(n)
    #           (V, P, R, gamma, prob_thresh, lever=1, prob='min', reduced=True)
    W = Bell_Robust(V, P, R, gamma, prob_thresh, lever, prob)
    its = 0
    while np.linalg.norm(V - W) > epsilon:
        its += 1
        V = W
        #              (V, P, R, gamma, prob_thresh, lever=1, prob='min', reduced=True)
        W = Bell_Robust(W, P, R, gamma, prob_thresh, lever, prob)
        if (
            gamma == 1 and max(abs(V)) > threshold
        ):  # threshold in case the value is actually infinity, used when gamma=1
            return V
    print("number of iterations: ", its)
    return W


# Robust expected Bellman operator
# ignore this one
def Bell_Robust_expected(V, P, R, gamma, prob="min", reduced=True):
    # inputs:
    # V: a vector storing the approximated value at each state under some policy
    # P: P[a,i,j] is the probability of transitioning from i to j when action a is taken
    # R: R[a,i,j] is the reward when going from i to j when action a is taken
    # m: number of actions
    # n: number of states
    m, n, n = P.shape
    res = np.zeros(n)
    v = V.copy()
    if prob == "min":
        for i in range(n):
            # if sum(P[0,i,:]) != 1:
            # print('state index: ', i, ', action index: 0', ', prob vec: ', P[0,i,:])
            res[i] = max(P[0, i, :] * (R[0, i, :] + gamma * v))
            for a in range(m):
                # if sum(P[a,i,:]) != 1:
                # print('state index: ', i, ', action index: ',a, ', prob vec: ', P[a,i,:])
                res[i] = min(res[i], max(P[a, i, :] * (R[a, i, :] + gamma * v)))
    if prob == "max":
        for i in range(n):
            # if sum(P[0,i,:]) != 1:
            # print('state index: ', i, ', action index: 0', ', prob vec: ', P[0,i,:])
            res[i] = min(P[0, i, :] * (R[0, i, :] + gamma * v))
            for a in range(m):
                # if sum(P[a,i,:]) != 1:
                # print('state index: ', i, ', action index: ',a, ', prob vec: ', P[a,i,:])
                res[i] = max(res[i], min(P[a, i, :] * (R[a, i, :] + gamma * v)))
    return res


# Value Iteration Robust expected (ignore)
def ValueIteration_Robust_expected(
    P, R, gamma=0.9, epsilon=10 ** (-10), prob="min", threshold=float("inf")
):
    m, n, n = P.shape
    V = np.zeros(n)
    W = Bell_Robust_expected(V, P, R, gamma, prob)
    its = 0
    # print(W)
    # print(np.linalg.norm(V-W))
    # print(np.linalg.norm(V-W) > epsilon)
    while np.linalg.norm(V - W) > epsilon:
        its += 1
        V = W
        W = Bell_Robust_expected(W, P, R, gamma, prob)
        if (
            gamma == 1 and max(abs(V)) > threshold
        ):  # threshold in case the value is actually infinity, used when gamma=1
            return V
    print("number of iterations: ", its)
    return W


# Evaluate a policy in an MDP. WARNING:works only with R[s]
# ???
def policy_value(mu, P, R, gamma=0.9, epsilon=10 ** (-10)):
    m, n, n = P.shape
    V = np.ones(n)
    W = V.copy()
    for s in range(n):
        V[s] = R[s] + gamma * sum(
            mu[s, a] * sum(P[a, s, sp] * V[sp] for sp in range(n)) for a in range(m)
        )
    while np.linalg.norm(V - W) > epsilon:
        W = V.copy()
        for s in range(n):
            V[s] = R[s] + gamma * sum(
                mu[s, a] * sum(P[a, s, sp] * V[sp] for sp in range(n)) for a in range(m)
            )
    return V


# Getting Policy from value function
def GetPolicy_Robust_expected(V, P, R, gamma, prob="min"):
    if len(R.shape) == 2:
        R = expand(R)
    m, n, l = P.shape
    # Vals[i][a] is the value of taking action a from state i, i.e. Q(i,a)
    if prob == "min":
        Vals = [
            [max(P[a, i, :] * (R[a, i, :] + gamma * V)) for a in range(m)]
            for i in range(n)
        ]
        pi = [np.argmin(Vals[i]) for i in range(n)]
    if prob == "max":
        Vals = [
            [min(P[a, i, :] * (R[a, i, :] + gamma * V)) for a in range(m)]
            for i in range(n)
        ]
        pi = [np.argmax(Vals[i]) for i in range(n)]
    return pi, Vals


# Getting Policy from value function
def GetPolicy_Robust(V, P, R, gamma, prob="min", lever=1, prob_thresh=0):
    if len(R.shape) == 2:
        R = expand(R)
    m, n, l = P.shape
    if prob == "min":
        Vals = [
            [
                lever
                * max(
                    np.array(R[a, i, np.where(P[a, i, :] > prob_thresh)[0]])
                    + gamma * np.array(V[np.where(P[a, i, :] > prob_thresh)[0]])
                )
                + (1 - lever)
                * min(
                    np.array(R[a, i, np.where(P[a, i, :] > prob_thresh)[0]])
                    + gamma * np.array(V[np.where(P[a, i, :] > prob_thresh)[0]])
                )
                for a in range(m)
            ]
            for i in range(n)
        ]
        pi = [np.argmin(Vals[i]) for i in range(n)]
    if prob == "max":
        Vals = [
            [
                lever
                * min(
                    np.array(R[a, i, np.where(P[a, i, :] > prob_thresh)[0]])
                    + gamma * np.array(V[np.where(P[a, i, :] > prob_thresh)[0]])
                )
                + (1 - lever)
                * max(
                    np.array(R[a, i, np.where(P[a, i, :] > prob_thresh)[0]])
                    + gamma * np.array(V[np.where(P[a, i, :] > prob_thresh)[0]])
                )
                for a in range(m)
            ]
            for i in range(n)
        ]
        pi = [np.argmax(Vals[i]) for i in range(n)]
    return pi, Vals


#################################################################


#################################################################
def SolveMDP(
    P,
    R,
    gamma=0.9,
    epsilon=10 ** (-10),
    p=True,
    prob="min",
    method="Value",
    threshold=float("inf"),
):
    # P: Transition probability
    # R: Reward matrix
    # epsilon: convergence param of value iteration
    # prob: Specify if the objective is to minimize ('min') or maxmize ('max') outcome
    m, n, n = P.shape
    if len(R.shape) < 3:
        R = expand(R, n, m)
    if method == "Value":
        print("Value iteration")
        V = np.array(ValueIteration(P, R, gamma, epsilon, prob, threshold))
        pi, Vals = GetPolicy(V, P, R, gamma, prob)

    if p:
        print("Optimal Value:", list(V))
        print("Optimal Policy:", list(pi))
        # print('Vals:', Vals)
    return list(V), list(pi), Vals


def SolveMDP_Robust_expected(
    P,
    R,
    gamma=0.9,
    epsilon=10 ** (-10),
    p=True,
    prob="min",
    method="Value",
    threshold=float("inf"),
):
    # P: Transition probability
    # R: Reward matrix
    # epsilon: convergence param of value iteration
    # prob: Specify if the objective is to minimize ('min') or maxmize ('max') outcome
    m, n, n = P.shape
    # V_init = np.array(R)
    if len(R.shape) < 3:
        R = expand(R, n, m)
    if method == "Value":
        print("Value iteration")
        V = np.array(
            ValueIteration_Robust_expected(P, R, gamma, epsilon, prob, threshold)
        )
        pi, Vals = GetPolicy_Robust_expected(V, P, R, gamma, prob)

    if p:
        print("Optimal Value:", list(V))
        print("Optimal Policy:", list(pi))
        # print('Vals:', Vals)
    return list(V), list(pi), Vals


def SolveMDP_Robust(
    P,
    R,
    lever=1,
    prob_thresh=0,
    gamma=0.9,
    epsilon=10 ** (-10),
    p=True,
    prob="min",
    method="Value",
    threshold=float("inf"),
):
    # P: Transition probability
    # R: Reward matrix
    # epsilon: convergence param of value iteration
    # prob: Specify if the objective is to minimize ('min') or maxmize ('max') outcome
    m, n, n = P.shape
    if len(R.shape) < 3:
        R = expand(R, n, m)
    if method == "Value":
        print("Value iteration")
        # ValueIteration_Robust(P, R, gamma=0.9, epsilon=10**(-10), prob='min', lever = 1, threshold=float('inf'), prob_thresh=0)
        V = np.array(
            ValueIteration_Robust(
                P, R, gamma, epsilon, prob, lever, threshold, prob_thresh
            )
        )
        #                   GetPolicy_Robust(V, P, R, gamma, prob='min', lever=1, prob_thresh=0)
        pi, Vals = GetPolicy_Robust(V, P, R, gamma, prob, lever, prob_thresh)

    if p:
        print("Optimal Value:", list(V))
        print("Optimal Policy:", list(pi))
        # print('Vals:', Vals)
    return list(V), list(pi), Vals


def SolveMDP_Robust_quantiles(
    P,
    R,
    percentile=50,
    prob_thresh=0,
    gamma=0.9,
    epsilon=10 ** (-10),
    p=True,
    prob="min",
    method="Value",
    threshold=float("inf"),
):
    # P: Transition probability
    # R: Reward matrix
    # epsilon: convergence param of value iteration
    # prob: Specify if the objective is to minimize ('min') or maxmize ('max') outcome
    m, n, n = P.shape
    if len(R.shape) < 3:
        R = expand(R, n, m)
    if method == "Value":
        print("Value iteration")
        #      ValueIteration_Robust_quantiles(P, R, gamma=0.9, epsilon=10**(-10), prob='min', percentile = 50, threshold=float('inf'), prob_thresh=0)
        V, Ind = ValueIteration_Robust_quantiles(
            P, R, gamma, epsilon, prob, percentile, threshold, prob_thresh
        )
        #                                           (V, P, R, Ind, gamma, prob='min', percentile=50, prob_thresh=0)
        pi, Vals = GetPolicy_Robust_quantiles(
            V, P, R, Ind, gamma, prob, percentile, prob_thresh
        )

    if p:
        print("Optimal Value:", list(V))
        print("Optimal Policy:", list(pi))
        # print('Vals:', Vals)
    return list(V), list(pi), Vals


def SolveMDP_ICTE(
    P,
    R,
    percentile=50,
    prob_thresh=0,
    gamma=0.9,
    epsilon=10 ** (-10),
    p=True,
    prob="min",
    method="Value",
    threshold=float("inf"),
):
    # P: Transition probability
    # R: Reward matrix
    # epsilon: convergence param of value iteration
    # prob: Specify if the objective is to minimize ('min') or maxmize ('max') outcome
    m, n, n = P.shape
    if len(R.shape) < 3:
        R = expand(R, n, m)
    if method == "Value":
        print("Value iteration")
        #      ValueIteration_Robust_quantiles(P, R, gamma=0.9, epsilon=10**(-10), prob='min', percentile = 50, threshold=float('inf'), prob_thresh=0)
        V, Vals_mat = ValueIteration_ICTE(
            P, R, gamma, epsilon, prob, percentile, threshold, prob_thresh
        )
        #                                           (V, P, R, Ind, gamma, prob='min', percentile=50, prob_thresh=0)
        pi, Vals = GetPolicy_ICTE(
            V, P, R, Vals_mat, gamma, prob, percentile, prob_thresh
        )

    if p:
        print("Optimal Value:", list(V))
        print("Optimal Policy:", list(pi))
        # print('Vals:', Vals)
    return list(V), list(pi), Vals


#################################################################
