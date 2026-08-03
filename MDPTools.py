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


def standardize_values(values):
    total = sum(values)
    standardized_values = [(x / total) for x in values]
    return standardized_values

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