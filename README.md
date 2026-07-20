# sepsis-fluids

Clustering and solving the MDP uses Apache Spark. 

# train.py

Clustering the patient-hours takes place in train.py.

INPUTS

features
[ID, TIME, ...features..., ACTION, RISK]
where ID is a string representing a patient, and TIME, ACTION, RISK are integers.

end_states
[ID, end_state, Reward]
where ID and end_state are strings, and Reward is an integer corresponding to the reward assigned to the end state. 

OUTPUTS
df_trained_out 
[ID, TIME, ...features..., ACTION, RISK, CLUSTER, NEXT_CLUSTER]
where CLUSTER is an integer representing the current cluster of the patient ID and time TIME, and NEXT_CLUSTER is the cluster of the patient ID and time TIME + 1.

R_df

P_df


# solveMDP.py

solveMDP takes in the outputs from train.py and solves for the optimal policy. 

INPUTS

R_df

P_df


OUTPUTS

pi_E_df

Q_E_df

# analyze_mrl_results.ipynb