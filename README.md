# sepsis-fluids


# train.py

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

# solveMDP.py

solveMDP.py takes in the outputs from train.py and solves for the optimal policy. 


INPUTS

R_df

P_df



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
represents the action-value function of the MDP. The value at row r and column c represents the value of action c when at cluster r. 

# analyze_mrl_results.ipynb

This is the python notebook used to generate figures to visualize results and evaluate the MDP model.

The cluster heatmap section generates a heatmap describing the average value of each feature in each clinical state relative to the overall average value in the dataset, like in figure 3 of the paper. 

