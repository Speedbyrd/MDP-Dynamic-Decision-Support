# MDP Dynamic Decision Support


# train_clusters(features, end_states)

Assigns clusters (clinical states) to the patient-hours.


INPUTS

features

```python
['ID', 'TIME', ...features..., 'ACTION', 'RISK']
```
where ID is a string representing a patient, and TIME, ACTION, RISK are integers.

end_states

```python
['ID', 'end_state', 'Reward']
```
where ID and end_state are strings, and Reward is an integer corresponding to the reward assigned to the end state. 


OUTPUTS

df_trained_out 

```python
['ID', 'TIME', ...features..., 'ACTION', 'RISK', 'CLUSTER', 'NEXT_CLUSTER']
```
where CLUSTER is an integer representing the current cluster of the patient ID and time TIME, and NEXT_CLUSTER is the cluster of the patient ID and time TIME + 1.

Note that the results as visualized the paper are re-adjusted to 1-indexing for readability, but the code numbers clusters starting from 0.

R_df
```python
['CLUSTER', 'RISK']
```

P_df
```python
['CLUSTER', 'ACTION', 'NEXT_CLUSTER', 'prob']
```
where prob is the probability of transitioning to NEXT_CLUSTER when taking action ACTION in CLUSTER. 

# solve_MDP(P_df, R_df)

Takes in the outputs from train_clusters and solves for the optimal policy. 


INPUTS

R_df

Output from train_clusters.

P_df

Output from train_clusters.



OUTPUTS

pi_E_df

```python
['CLUSTER', 0]
```
values in the column 0 represents the optimal action found by the solver for CLUSTER

Q_E_df
```python
[...actions...]
```
represents the action-value function (long-term risk) of the MDP. The value at row r and column c represents the value of action c when at cluster r. 

# simulate_trajectories(df_trained_out, P_df, pi_E_df, end_states, max_steps, random_seed)

Simulates trajectories for the set of patients in df_trained_out. Starts out at the first known state of each patient and always chooses the action according to the policy as described in pi_E_df, where after each action the next state of the patient is selected at random based on the transition probabilities in P_df. 


INPUTS

df_trained_out

Output from train_clusters. 

P_df

The transition probability output from train_clusters

pi_E_df

The optimal policy output from solve_MDP


max_steps

The maximum number of time steps to simulate for. Defaults to 5000.

random_seed

A random seed determining the transitions that can be set for reproducable results. Defaults to 42


OUTPUTS

result_E

```python
['ID', 'TIME', 'CLUSTER', 'ACTION']
```
For each patient ID, the CLUSTER at TIME 0 is the (actual) first state they began in. Each subsequent CLUSTER is the simulated state transitioned to from the previous ACTION.


# analyze_mrl_results.ipynb

This is the python notebook used to generate figures to visualize results and evaluate the MDP model.

The cluster heatmap section generates a heatmap describing the average value of each feature in each clinical state relative to the overall average value in the dataset, like in figure 3 of the paper. 

The action values and frequencies section visualizes the frequency of each action and the long-term risk of each action, for each cluster (state). An example of this can be seen in figure 4 of the paper.

The absorbing probabilities section can be used to show the probability of reaching each end state when starting from a given cluster and taking a given action, and then following the optimal policy thereafter. An example of this can be seen in figure 4 of the paper, where we showed the probability of reaching each end state for state 5, action 3.