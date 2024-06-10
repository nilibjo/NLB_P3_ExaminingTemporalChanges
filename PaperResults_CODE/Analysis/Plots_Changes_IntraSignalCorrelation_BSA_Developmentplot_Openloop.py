import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pingouin as pgn


# Load Carotid Data

df_meas_tono = pd.read_pickle('Outputs_measurements_tono_BSA_WF.pkl')
df_mod_tono = pd.read_pickle('Outputs_models_tono_Open_BSA_WF.pkl')

# Load Finger Pressure Data

df_meas_FP = pd.read_pickle('Outputs_measurements_FP_BSA_WF.pkl')
df_mod_FP = pd.read_pickle('Outputs_models_FP_Open_BSA_WF.pkl')

# Check for invalid E_max in finger pressure measurement derived data
for idx, row in df_meas_FP.iterrows():
    if (row["E_max"] > 20.) or (row["E_max"] < 0.):
        print(row["id"],idx)

print("Search complete")


# Split dataset by measurement day

df_meas_tono_pre = df_meas_tono[df_meas_tono['test_day'] == 'Pre']
df_meas_tono_mid = df_meas_tono[df_meas_tono['test_day'] == 'Mid']
df_meas_tono_post = df_meas_tono[df_meas_tono['test_day'] == 'Post']

df_meas_fp_pre = df_meas_FP[df_meas_FP['test_day'] == 'Pre']
df_meas_fp_mid = df_meas_FP[df_meas_FP['test_day'] == 'Mid']
df_meas_fp_post = df_meas_FP[df_meas_FP['test_day'] == 'Post']

df_mod_tono_pre = df_mod_tono[df_mod_tono['test_day'] == 'Pre']
df_mod_tono_mid = df_mod_tono[df_mod_tono['test_day'] == 'Mid']
df_mod_tono_post = df_mod_tono[df_mod_tono['test_day'] == 'Post']

df_mod_fp_pre = df_mod_FP[df_mod_FP['test_day'] == 'Pre']
df_mod_fp_mid = df_mod_FP[df_mod_FP['test_day'] == 'Mid']
df_mod_fp_post = df_mod_FP[df_mod_FP['test_day'] == 'Post']

# Identify participants measured at each measurement day for carotid pressure
partid_pre = df_meas_tono_pre['partid']
partid_mid = df_meas_tono_mid['partid']
partid_post = df_meas_tono_post['partid']

print(partid_pre)
print(partid_mid)
print(partid_post)

# Find IDs that have measurements at all measurement days for carotid pressure
common_ids = []
for elt in partid_pre:
    if (elt in list(partid_mid)) and (elt in list(partid_post)):
        common_ids.append(elt)

common_ids_tono = common_ids.copy()

print(common_ids_tono)
print(common_ids)

# Identify participants measured at each measurement day for finger pressure
partid_pre = df_meas_fp_pre['partid']
partid_mid = df_meas_fp_mid['partid']
partid_post = df_meas_fp_post['partid']

common_ids = []
for elt in partid_pre:
    if (elt in list(partid_mid)) and (elt in list(partid_post)):
        common_ids.append(elt)

common_ids_fp = common_ids.copy()
print(common_ids_fp)


# Set up data structures for storing the computed changes
change_col = ['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]

change_col_D = ['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk", "R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET", "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET", "R_sys_12wk_CPET", "C_ao_12wk_CPET", "E_max_12wk_CPET", "R_sys_12wk_CPETtopre", "C_ao_12wk_CPETtopre", "E_max_12wk_CPETtopre"]

plot_col=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"]
plot_col_D=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3", "R1pC", "C1pC", "E1pC", "R3pC", "C3pC", "E3pC"]
df_meas_plot_C = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"])
df_meas_plot_D = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3", "R1pC", "C1pC", "E1pC", "R3pC", "C3pC", "E3pC"])
df_mod_plot_C = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"])
df_mod_plot_D = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3", "R1pC", "C1pC", "E1pC", "R3pC", "C3pC", "E3pC"])

df_meas_changes_tono = pd.DataFrame(columns=change_col)
df_mod_changes_tono = pd.DataFrame(columns=change_col)



# Load finger pressure data for the post CPET measurements
df_meas_FP_CPET = pd.read_pickle('Outputs_measurements_FP_BSA_CPET.pkl')
df_mod_FP_CPET = pd.read_pickle('Outputs_models_FP_Open_BSA_CPET.pkl')


# Split dataset into the relevant measurement sets
df_meas_fp_CPET_pre = df_meas_FP_CPET[df_meas_FP_CPET['test_day'] == '1CPET']
df_meas_fp_CPET_post = df_meas_FP_CPET[df_meas_FP_CPET['test_day'] == '3CPET'] 

df_mod_fp_CPET_pre = df_mod_FP_CPET[df_mod_FP_CPET['test_day'] == '1CPET']
df_mod_fp_CPET_post = df_mod_FP_CPET[df_mod_FP_CPET['test_day'] == '3CPET'] 

CPET_pre_ids = list(df_mod_fp_CPET_pre["partid"])
CPET_post_ids = list(df_mod_fp_CPET_post["partid"])


### LOOP THROUGH CAROTID MEASUREMENTS TO COMPUTE CHANGES
for elt in common_ids_tono:
    # Meas
    P_sys_6wk = df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['P_sys'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['P_dia'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['SV'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['R_sys'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['C_ao'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['E_max'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['E_max'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['SV'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['P_sys'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['P_dia'].iloc[0] - df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['P_dia'].iloc[0]

    E_max_612wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['E_max'].iloc[0] - df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['SV'].iloc[0] - df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['P_sys'].iloc[0] - df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_meas_tono_post[df_meas_tono_post['partid'] == elt]['P_dia'].iloc[0] - df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['P_dia'].iloc[0]
    
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Mod
    P_sys_6wk = df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['P_sys'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['P_dia'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['SV'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['R_sys'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['C_ao'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['E_max'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['E_max'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['SV'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['P_sys'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['P_dia'].iloc[0] - df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['P_dia'].iloc[0]

    E_max_612wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['E_max'].iloc[0] - df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['SV'].iloc[0] - df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['P_sys'].iloc[0] - df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_mod_tono_post[df_mod_tono_post['partid'] == elt]['P_dia'].iloc[0] - df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['P_dia'].iloc[0]
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Append
    df_meas_changes_tono = df_meas_changes_tono.append(dict(zip(change_col,line_vec_meas)),ignore_index=True)
    df_mod_changes_tono = df_mod_changes_tono.append(dict(zip(change_col,line_vec_mod)),ignore_index=True)

    plot_meas_C_vec = [elt, df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['R_sys'].iloc[0], df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['R_sys'].iloc[0], df_meas_tono_post[df_meas_tono_post['partid'] == elt]['R_sys'].iloc[0], df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['C_ao'].iloc[0], df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['C_ao'].iloc[0], df_meas_tono_post[df_meas_tono_post['partid'] == elt]['C_ao'].iloc[0],  df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['E_max'].iloc[0], df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['E_max'].iloc[0], df_meas_tono_post[df_meas_tono_post['partid'] == elt]['E_max'].iloc[0]]
    plot_mod_C_vec = [elt, df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['R_sys'].iloc[0], df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['R_sys'].iloc[0], df_mod_tono_post[df_mod_tono_post['partid'] == elt]['R_sys'].iloc[0], df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['C_ao'].iloc[0], df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['C_ao'].iloc[0], df_mod_tono_post[df_mod_tono_post['partid'] == elt]['C_ao'].iloc[0],  df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['E_max'].iloc[0], df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['E_max'].iloc[0], df_mod_tono_post[df_mod_tono_post['partid'] == elt]['E_max'].iloc[0]]
    
    df_meas_plot_C = df_meas_plot_C.append(dict(zip(plot_col, plot_meas_C_vec)), ignore_index=True)
    df_mod_plot_C = df_mod_plot_C.append(dict(zip(plot_col, plot_mod_C_vec)), ignore_index=True)
     
    


# Repeat for finger pressure (fp)
df_meas_changes_fp = pd.DataFrame(columns=change_col)
df_mod_changes_fp = pd.DataFrame(columns=change_col)

### LOOP THROUGH FINGER PRESSURE MEASUREMENTS TO COMPUTE CHANGES
for elt in common_ids_fp:
    # Meas
    P_sys_6wk = df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['P_sys'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['P_dia'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['SV'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['SV'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['P_sys'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['P_dia'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['P_dia'].iloc[0]

    E_max_612wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['SV'].iloc[0] - df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['P_sys'].iloc[0] - df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_meas_fp_post[df_meas_fp_post['partid'] == elt]['P_dia'].iloc[0] - df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['P_dia'].iloc[0]
    
    if elt in CPET_pre_ids:
        CPET_R_sys_pre = df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_pre = df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_pre = df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['C_ao'].iloc[0]
        print(CPET_R_sys_pre)
        CPET_R_sys_1_meas = df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_1_meas = df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_1_meas = df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0]
    else:
        CPET_R_sys_pre = np.nan
        CPET_E_max_pre = np.nan
        CPET_C_ao_pre = np.nan
        CPET_R_sys_1_meas = np.nan
        CPET_E_max_1_meas = np.nan
        CPET_C_ao_1_meas = np.nan

    if elt in CPET_post_ids:
        CPET_R_sys_12wk = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_12wk = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_12wk = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0] 
        CPET_R_sys_post = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_post[df_meas_fp_post['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_post = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_post[df_meas_fp_post['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_post = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_post[df_meas_fp_post['partid'] == elt]['C_ao'].iloc[0]
        CPET_R_sys_2_meas = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] 
        CPET_E_max_2_meas = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] 
        CPET_C_ao_2_meas = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] 
        CPET_R_sys_12wk_topre = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_12wk_topre = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_C_ao_12wk_topre = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0]
    else:
        CPET_R_sys_post = np.nan
        CPET_C_ao_post = np.nan
        CPET_E_max_post = np.nan
        CPET_R_sys_12wk = np.nan
        CPET_C_ao_12wk = np.nan
        CPET_E_max_12wk = np.nan
        CPET_R_sys_2_meas = np.nan
        CPET_E_max_2_meas = np.nan
        CPET_C_ao_2_meas = np.nan
        CPET_R_sys_12wk_topre = np.nan
        CPET_E_max_12wk_topre = np.nan
        CPET_C_ao_12wk_topre = np.nan
    
    
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                          R_sys_6wk, C_ao_6wk, E_max_6wk, 
                          P_sys_12wk, P_dia_12wk, SV_12wk, 
                          R_sys_12wk, C_ao_12wk, E_max_12wk, 
                          P_sys_612wk, P_dia_612wk, SV_612wk, 
                          R_sys_612wk, C_ao_612wk, E_max_612wk, 
                          CPET_R_sys_pre, CPET_C_ao_pre, CPET_E_max_pre, 
                          CPET_R_sys_post, CPET_C_ao_post, CPET_E_max_post, 
                          CPET_R_sys_12wk, CPET_C_ao_12wk, CPET_E_max_12wk, 
                          CPET_R_sys_12wk_topre, CPET_C_ao_12wk_topre, CPET_E_max_12wk_topre]
    
    # Mod
    P_sys_6wk = df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['P_sys'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['P_dia'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['SV'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['SV'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['P_sys'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['P_dia'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['P_dia'].iloc[0]

    E_max_612wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['SV'].iloc[0] - df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['P_sys'].iloc[0] - df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_mod_fp_post[df_mod_fp_post['partid'] == elt]['P_dia'].iloc[0] - df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['P_dia'].iloc[0]
    
    if elt in CPET_pre_ids:
        CPET_R_sys_pre = df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_pre = df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_pre = df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['C_ao'].iloc[0]
        CPET_R_sys_1_mod = df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_1_mod = df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_1_mod = df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0]
    else:
        CPET_R_sys_pre = np.nan
        CPET_E_max_pre = np.nan
        CPET_C_ao_pre = np.nan
        CPET_R_sys_1_mod = np.nan
        CPET_E_max_1_mod = np.nan
        CPET_C_ao_1_mod = np.nan
    
    if elt in CPET_post_ids:
        CPET_R_sys_12wk = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_12wk = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_12wk = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0] 
        CPET_R_sys_post = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_post[df_mod_fp_post['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_post = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_post[df_mod_fp_post['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_post = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_post[df_mod_fp_post['partid'] == elt]['C_ao'].iloc[0]
        CPET_R_sys_2_mod = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] 
        CPET_E_max_2_mod = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] 
        CPET_C_ao_2_mod = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] 
        CPET_R_sys_12wk_topre = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_12wk_topre = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_C_ao_12wk_topre = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0]
    else:
        CPET_R_sys_post = np.nan
        CPET_C_ao_post = np.nan
        CPET_E_max_post = np.nan
        CPET_R_sys_12wk = np.nan
        CPET_C_ao_12wk = np.nan
        CPET_E_max_12wk = np.nan
        CPET_R_sys_2_mod = np.nan
        CPET_E_max_2_mod = np.nan
        CPET_C_ao_2_mod = np.nan
        CPET_R_sys_12wk_topre = np.nan
        CPET_E_max_12wk_topre = np.nan
        CPET_C_ao_12wk_topre = np.nan
    
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                         R_sys_6wk, C_ao_6wk, E_max_6wk, 
                         P_sys_12wk, P_dia_12wk, SV_12wk, 
                         R_sys_12wk, C_ao_12wk, E_max_12wk, 
                         P_sys_612wk, P_dia_612wk, SV_612wk, 
                         R_sys_612wk, C_ao_612wk, E_max_612wk, 
                         CPET_R_sys_pre, CPET_C_ao_pre, CPET_E_max_pre, 
                         CPET_R_sys_post, CPET_C_ao_post, CPET_E_max_post, 
                         CPET_R_sys_12wk, CPET_C_ao_12wk, CPET_E_max_12wk, 
                         CPET_R_sys_12wk_topre, CPET_C_ao_12wk_topre, CPET_E_max_12wk_topre]
    
    # Append
    df_meas_changes_fp = df_meas_changes_fp.append(dict(zip(change_col_D,line_vec_meas)),ignore_index=True)
    df_mod_changes_fp = df_mod_changes_fp.append(dict(zip(change_col_D,line_vec_mod)),ignore_index=True)
    
    plot_meas_D_vec = [elt, df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0], df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['R_sys'].iloc[0], df_meas_fp_post[df_meas_fp_post['partid'] == elt]['R_sys'].iloc[0], df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['C_ao'].iloc[0], df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['C_ao'].iloc[0], df_meas_fp_post[df_meas_fp_post['partid'] == elt]['C_ao'].iloc[0],  df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['E_max'].iloc[0], df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['E_max'].iloc[0], df_meas_fp_post[df_meas_fp_post['partid'] == elt]['E_max'].iloc[0], CPET_R_sys_1_meas, CPET_C_ao_1_meas, CPET_E_max_1_meas, CPET_R_sys_2_meas, CPET_C_ao_2_meas, CPET_E_max_2_meas]
    plot_mod_D_vec = [elt, df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0], df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['R_sys'].iloc[0], df_mod_fp_post[df_mod_fp_post['partid'] == elt]['R_sys'].iloc[0], df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['C_ao'].iloc[0], df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['C_ao'].iloc[0], df_mod_fp_post[df_mod_fp_post['partid'] == elt]['C_ao'].iloc[0],  df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['E_max'].iloc[0], df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['E_max'].iloc[0], df_mod_fp_post[df_mod_fp_post['partid'] == elt]['E_max'].iloc[0], CPET_R_sys_1_mod, CPET_C_ao_1_mod, CPET_E_max_1_mod, CPET_R_sys_2_mod, CPET_C_ao_2_mod, CPET_E_max_2_mod]
    
    df_meas_plot_D = df_meas_plot_D.append(dict(zip(plot_col_D, plot_meas_D_vec)), ignore_index=True)
    df_mod_plot_D = df_mod_plot_D.append(dict(zip(plot_col_D, plot_mod_D_vec)), ignore_index=True)



### Plot changes  

# Plot Mod D CHANGES
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(6,10), sharex='col')

ax1.plot([0.5, 2.5], [0, 0], 'k')
ax2.plot([0.5, 2.5], [0, 0], 'k')
ax3.plot([0.5, 2.5], [0, 0], 'k')
ax1.boxplot( [ df_mod_changes_tono["R_sys_6wk"],df_mod_changes_tono["R_sys_12wk"] ] )
ax2.boxplot( [ df_mod_changes_tono["C_ao_6wk"],df_mod_changes_tono["C_ao_12wk"] ] )
ax3.boxplot( [ df_mod_changes_tono["E_max_6wk"],df_mod_changes_tono["E_max_12wk"] ] )

plt.sca(ax1)
plt.xticks([1,2], ["6", "12"], fontsize=12)

plt.ylabel("$R_{\mathrm{sys}}$ [mmHg s/(mL $\mathrm{m}^2$)]", fontsize=12)
plt.sca(ax2)
plt.xticks([1,2], ["6", "12"], fontsize=12)

plt.ylabel("$C_{\mathrm{ao}}$ [mL/(mmHg $\mathrm{m}^2$)]", fontsize=12)
plt.sca(ax3)
plt.xticks([1,2], ["6", "12"], fontsize=12)
plt.xlabel("Weeks", fontsize=12)
plt.ylabel("$E_{\mathrm{max}}$ [mmHg/(mL $\mathrm{m}^2$)]", fontsize=12)

plt.savefig("CorrelationChanges_Means/ModC_CHANGE_boxplots_Open.pdf")
plt.show()

# Plot Mod D CHANGES
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(6,10), sharex='col')

ax1.plot([0.5, 4.5], [0, 0], 'k')
ax2.plot([0.5, 4.5], [0, 0], 'k')
ax3.plot([0.5, 4.5], [0, 0], 'k')
ax1.boxplot( [ df_mod_changes_fp["R_sys_pre_CPET"].dropna(),df_mod_changes_fp["R_sys_6wk"],df_mod_changes_fp["R_sys_12wk"],df_mod_changes_fp["R_sys_12wk_CPETtopre"].dropna() ] )
ax2.boxplot( [ df_mod_changes_fp["C_ao_pre_CPET"].dropna(),df_mod_changes_fp["C_ao_6wk"],df_mod_changes_fp["C_ao_12wk"],df_mod_changes_fp["C_ao_12wk_CPETtopre"].dropna() ] )
ax3.boxplot( [ df_mod_changes_fp["E_max_pre_CPET"].dropna(),df_mod_changes_fp["E_max_6wk"],df_mod_changes_fp["E_max_12wk"],df_mod_changes_fp["E_max_12wk_CPETtopre"].dropna() ] )

plt.sca(ax1)
plt.xticks([1,2,3,4], ["0*", "6", "12", "12*"], fontsize=12)
#plt.xlabel("Weeks", fontsize=12)
plt.ylabel("$R_{\mathrm{sys}}$ [mL s/(mmHg $\mathrm{m}^2$)]", fontsize=12)
plt.sca(ax2)
plt.xticks([1,2,3,4], ["0*", "6", "12", "12*"], fontsize=12)
#plt.xlabel("Weeks", fontsize=12)
plt.ylabel("$C_{\mathrm{ao}}$ [mL/(mmHg $\mathrm{m}^2$)]", fontsize=12)
plt.sca(ax3)
plt.xticks([1,2,3,4], ["0*", "6", "12", "12*"], fontsize=12)
plt.xlabel("Weeks", fontsize=12)
plt.ylabel("$E_{\mathrm{max}}$ [mmHg/(mL $\mathrm{m}^2$)]", fontsize=12)

plt.savefig("CorrelationChanges_Means/ModD_CHANGE_boxplots_Open.pdf")
plt.show()


