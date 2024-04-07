
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pingouin as pgn


# Load Carotid Data

df_meas_tono = pd.read_pickle('Outputs_measurements_tono_BSA_WF.pkl')
df_mod_tono = pd.read_pickle('Outputs_models_tono_BSA_WF.pkl')

df_mod_tono_open = pd.read_pickle('Outputs_models_tono_Open_BSA_WF.pkl')

# Load Finger Data

df_meas_FP = pd.read_pickle('Outputs_measurements_FP_BSA_WF.pkl')
df_mod_FP = pd.read_pickle('Outputs_models_FP_BSA_WF.pkl')

df_mod_FP_open = pd.read_pickle('Outputs_models_FP_Open_BSA_WF.pkl')

# Check for invalid E_max measurement among finger pressure results
for idx, row in df_meas_FP.iterrows():
    if (row["E_max"] > 20.) or (row["E_max"] < 0.):
        print("Invalid E_max at (ID,idx):",row["id"],idx)

print("Search complete")


# Split Dataset by measurement day

df_meas_tono_pre = df_meas_tono[df_meas_tono['test_day'] == 'Pre-test day 2']
df_meas_tono_mid = df_meas_tono[df_meas_tono['test_day'] == 'Mid-test']
df_meas_tono_post = df_meas_tono[df_meas_tono['test_day'] == 'Post-test day 2']

df_meas_fp_pre = df_meas_FP[df_meas_FP['test_day'] == 'Pre-test day 2']
df_meas_fp_mid = df_meas_FP[df_meas_FP['test_day'] == 'Mid-test']
df_meas_fp_post = df_meas_FP[df_meas_FP['test_day'] == 'Post-test day 2']

df_mod_tono_pre = df_mod_tono[df_mod_tono['test_day'] == 'Pre-test day 2']
df_mod_tono_mid = df_mod_tono[df_mod_tono['test_day'] == 'Mid-test']
df_mod_tono_post = df_mod_tono[df_mod_tono['test_day'] == 'Post-test day 2']

df_mod_fp_pre = df_mod_FP[df_mod_FP['test_day'] == 'Pre-test day 2']
df_mod_fp_mid = df_mod_FP[df_mod_FP['test_day'] == 'Mid-test']
df_mod_fp_post = df_mod_FP[df_mod_FP['test_day'] == 'Post-test day 2']

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

# Compute changes
change_col = ['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]

df_meas_changes_tono = pd.DataFrame(columns=change_col)
df_mod_changes_tono = pd.DataFrame(columns=change_col)

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

# Repeat for finger pressure (fp)
df_meas_changes_fp = pd.DataFrame(columns=change_col)
df_mod_changes_fp = pd.DataFrame(columns=change_col)

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
    
    #['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
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
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Append
    df_meas_changes_fp = df_meas_changes_fp.append(dict(zip(change_col,line_vec_meas)),ignore_index=True)
    df_mod_changes_fp = df_mod_changes_fp.append(dict(zip(change_col,line_vec_mod)),ignore_index=True)


# Compute correlation values between paired subsets of the data for the closed-loop model

# 6, 12
Rsys_6_12_me_C = pgn.corr(df_meas_changes_tono["R_sys_6wk"], df_meas_changes_tono["R_sys_12wk"])

Cao_6_12_me_C = pgn.corr(df_meas_changes_tono["C_ao_6wk"], df_meas_changes_tono["C_ao_12wk"])

Emax_6_12_me_C = pgn.corr(df_meas_changes_tono["E_max_6wk"], df_meas_changes_tono["E_max_12wk"])


Rsys_6_12_me_D = pgn.corr(df_meas_changes_fp["R_sys_6wk"], df_meas_changes_fp["R_sys_12wk"])
Cao_6_12_me_D = pgn.corr(df_meas_changes_fp["C_ao_6wk"], df_meas_changes_fp["C_ao_12wk"])
Emax_6_12_me_D = pgn.corr(df_meas_changes_fp["E_max_6wk"], df_meas_changes_fp["E_max_12wk"])

Rsys_6_12_C = pgn.corr(df_mod_changes_tono["R_sys_6wk"], df_mod_changes_tono["R_sys_12wk"])
Cao_6_12_C = pgn.corr(df_mod_changes_tono["C_ao_6wk"], df_mod_changes_tono["C_ao_12wk"])
Emax_6_12_C = pgn.corr(df_mod_changes_tono["E_max_6wk"], df_mod_changes_tono["E_max_12wk"])

Rsys_6_12_D = pgn.corr(df_mod_changes_fp["R_sys_6wk"], df_mod_changes_fp["R_sys_12wk"])
Cao_6_12_D = pgn.corr(df_mod_changes_fp["C_ao_6wk"], df_mod_changes_fp["C_ao_12wk"])
Emax_6_12_D = pgn.corr(df_mod_changes_fp["E_max_6wk"], df_mod_changes_fp["E_max_12wk"])

# 6, 6-12
Rsys_6_612_me_C = pgn.corr(df_meas_changes_tono["R_sys_6wk"], df_meas_changes_tono["R_sys_612wk"])
Cao_6_612_me_C = pgn.corr(df_meas_changes_tono["C_ao_6wk"], df_meas_changes_tono["C_ao_612wk"])
Emax_6_612_me_C = pgn.corr(df_meas_changes_tono["E_max_6wk"], df_meas_changes_tono["E_max_612wk"])

Rsys_6_612_me_D = pgn.corr(df_meas_changes_fp["R_sys_6wk"], df_meas_changes_fp["R_sys_612wk"])
Cao_6_612_me_D = pgn.corr(df_meas_changes_fp["C_ao_6wk"], df_meas_changes_fp["C_ao_612wk"])
Emax_6_612_me_D = pgn.corr(df_meas_changes_fp["E_max_6wk"], df_meas_changes_fp["E_max_612wk"])

Rsys_6_612_C = pgn.corr(df_mod_changes_tono["R_sys_6wk"], df_mod_changes_tono["R_sys_612wk"])
Cao_6_612_C = pgn.corr(df_mod_changes_tono["C_ao_6wk"], df_mod_changes_tono["C_ao_612wk"])
Emax_6_612_C = pgn.corr(df_mod_changes_tono["E_max_6wk"], df_mod_changes_tono["E_max_612wk"])

Rsys_6_612_D = pgn.corr(df_mod_changes_fp["R_sys_6wk"], df_mod_changes_fp["R_sys_612wk"])
Cao_6_612_D = pgn.corr(df_mod_changes_fp["C_ao_6wk"], df_mod_changes_fp["C_ao_612wk"])
Emax_6_612_D = pgn.corr(df_mod_changes_fp["E_max_6wk"], df_mod_changes_fp["E_max_612wk"])

# 6-12, 12
Rsys_612_12_me_C = pgn.corr(df_meas_changes_tono["R_sys_612wk"], df_meas_changes_tono["R_sys_12wk"])
Cao_612_12_me_C = pgn.corr(df_meas_changes_tono["C_ao_612wk"], df_meas_changes_tono["C_ao_12wk"])
Emax_612_12_me_C = pgn.corr(df_meas_changes_tono["E_max_612wk"], df_meas_changes_tono["E_max_12wk"])

Rsys_612_12_me_D = pgn.corr(df_meas_changes_fp["R_sys_612wk"], df_meas_changes_fp["R_sys_12wk"])
Cao_612_12_me_D = pgn.corr(df_meas_changes_fp["C_ao_612wk"], df_meas_changes_fp["C_ao_12wk"])
Emax_612_12_me_D = pgn.corr(df_meas_changes_fp["E_max_612wk"], df_meas_changes_fp["E_max_12wk"])

Rsys_612_12_C = pgn.corr(df_mod_changes_tono["R_sys_612wk"], df_mod_changes_tono["R_sys_12wk"])
Cao_612_12_C = pgn.corr(df_mod_changes_tono["C_ao_612wk"], df_mod_changes_tono["C_ao_12wk"])
Emax_612_12_C = pgn.corr(df_mod_changes_tono["E_max_612wk"], df_mod_changes_tono["E_max_12wk"])

Rsys_612_12_D = pgn.corr(df_mod_changes_fp["R_sys_612wk"], df_mod_changes_fp["R_sys_12wk"])
Cao_612_12_D = pgn.corr(df_mod_changes_fp["C_ao_612wk"], df_mod_changes_fp["C_ao_12wk"])
Emax_612_12_D = pgn.corr(df_mod_changes_fp["E_max_612wk"], df_mod_changes_fp["E_max_12wk"])



m6_Rsys_C = pgn.corr(df_meas_changes_tono["R_sys_6wk"], df_mod_changes_tono["R_sys_6wk"])
m6_Cao_C = pgn.corr(df_meas_changes_tono["C_ao_6wk"], df_mod_changes_tono["C_ao_6wk"])
m6_Emax_C = pgn.corr(df_meas_changes_tono["E_max_6wk"], df_mod_changes_tono["E_max_6wk"])

m12_Rsys_C = pgn.corr(df_meas_changes_tono["R_sys_12wk"], df_mod_changes_tono["R_sys_12wk"])
m12_Cao_C = pgn.corr(df_meas_changes_tono["C_ao_12wk"], df_mod_changes_tono["C_ao_12wk"])
m12_Emax_C = pgn.corr(df_meas_changes_tono["E_max_12wk"], df_mod_changes_tono["E_max_12wk"])

m612_Rsys_C = pgn.corr(df_meas_changes_tono["R_sys_612wk"], df_mod_changes_tono["R_sys_612wk"])
m612_Cao_C = pgn.corr(df_meas_changes_tono["C_ao_612wk"], df_mod_changes_tono["C_ao_612wk"])
m612_Emax_C = pgn.corr(df_meas_changes_tono["E_max_612wk"], df_mod_changes_tono["E_max_612wk"])

m6_Rsys_D = pgn.corr(df_meas_changes_fp["R_sys_6wk"], df_mod_changes_fp["R_sys_6wk"])
m6_Cao_D = pgn.corr(df_meas_changes_fp["C_ao_6wk"], df_mod_changes_fp["C_ao_6wk"])
m6_Emax_D = pgn.corr(df_meas_changes_fp["E_max_6wk"], df_mod_changes_fp["E_max_6wk"])

m12_Rsys_D = pgn.corr(df_meas_changes_fp["R_sys_12wk"], df_mod_changes_fp["R_sys_12wk"])
m12_Cao_D = pgn.corr(df_meas_changes_fp["C_ao_12wk"], df_mod_changes_fp["C_ao_12wk"])
m12_Emax_D = pgn.corr(df_meas_changes_fp["E_max_12wk"], df_mod_changes_fp["E_max_12wk"])

m612_Rsys_D = pgn.corr(df_meas_changes_fp["R_sys_612wk"], df_mod_changes_fp["R_sys_612wk"])
m612_Cao_D = pgn.corr(df_meas_changes_fp["C_ao_612wk"], df_mod_changes_fp["C_ao_612wk"])
m612_Emax_D = pgn.corr(df_meas_changes_fp["E_max_612wk"], df_mod_changes_fp["E_max_612wk"])


################################################################################
### TREAT OPEN-LOOP RESULTS
################################################################################

# Compute changes in the same manner as for the open loop model

df_meas_tono_pre_open = df_meas_tono[df_meas_tono['test_day'] == 'Pre-test day 2']
df_meas_tono_mid_open = df_meas_tono[df_meas_tono['test_day'] == 'Mid-test']
df_meas_tono_post_open = df_meas_tono[df_meas_tono['test_day'] == 'Post-test day 2']

df_meas_fp_pre_open = df_meas_FP[df_meas_FP['test_day'] == 'Pre-test day 2']
df_meas_fp_mid_open = df_meas_FP[df_meas_FP['test_day'] == 'Mid-test']
df_meas_fp_post_open = df_meas_FP[df_meas_FP['test_day'] == 'Post-test day 2']

df_mod_tono_pre_open = df_mod_tono_open[df_mod_tono_open['test_day'] == 'Pre-test day 2']
df_mod_tono_mid_open = df_mod_tono_open[df_mod_tono_open['test_day'] == 'Mid-test']
df_mod_tono_post_open = df_mod_tono_open[df_mod_tono_open['test_day'] == 'Post-test day 2']

df_mod_fp_pre_open = df_mod_FP_open[df_mod_FP_open['test_day'] == 'Pre-test day 2']
df_mod_fp_mid_open = df_mod_FP_open[df_mod_FP_open['test_day'] == 'Mid-test']
df_mod_fp_post_open = df_mod_FP_open[df_mod_FP_open['test_day'] == 'Post-test day 2']

partid_pre_open = df_meas_tono_pre_open['partid']
partid_mid_open = df_meas_tono_mid_open['partid']
partid_post_open = df_meas_tono_post_open['partid']

print(partid_pre_open)
print(partid_mid_open)
print(partid_post_open)

common_ids_open = []
for elt in partid_pre_open:
    if (elt in list(partid_mid_open)) and (elt in list(partid_post_open)):
        common_ids_open.append(elt)

common_ids_tono_open = common_ids_open.copy()

print(common_ids_tono_open)
print(common_ids_open)

partid_pre_open = df_meas_fp_pre_open['partid']
partid_mid_open = df_meas_fp_mid_open['partid']
partid_post_open = df_meas_fp_post_open['partid']

common_ids_open = []
for elt in partid_pre_open:
    if (elt in list(partid_mid_open)) and (elt in list(partid_post_open)):
        common_ids_open.append(elt)

common_ids_fp_open = common_ids_open.copy()
print(common_ids_fp_open)

# Compute changes
change_col = ['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk",'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]

df_meas_changes_tono_open = pd.DataFrame(columns=change_col)
df_mod_changes_tono_open = pd.DataFrame(columns=change_col)

for elt in common_ids_tono_open:
    # Meas
    P_sys_6wk = df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['P_sys'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['P_dia'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['SV'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['R_sys'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['C_ao'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['E_max'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['E_max'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['C_ao'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['R_sys'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['SV'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['P_sys'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['P_dia'].iloc[0] - df_meas_tono_pre_open[df_meas_tono_pre_open['partid'] == elt]['P_dia'].iloc[0]

    E_max_612wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['E_max'].iloc[0] - df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['C_ao'].iloc[0] - df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['R_sys'].iloc[0] - df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['SV'].iloc[0] - df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['P_sys'].iloc[0] - df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_meas_tono_post_open[df_meas_tono_post_open['partid'] == elt]['P_dia'].iloc[0] - df_meas_tono_mid_open[df_meas_tono_mid_open['partid'] == elt]['P_dia'].iloc[0]
    
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Mod
    P_sys_6wk = df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['P_sys'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['P_dia'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['SV'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['R_sys'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['C_ao'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['E_max'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['E_max'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['C_ao'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['R_sys'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['SV'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['P_sys'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['P_dia'].iloc[0] - df_mod_tono_pre_open[df_mod_tono_pre_open['partid'] == elt]['P_dia'].iloc[0]
    
    E_max_612wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['E_max'].iloc[0] - df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['C_ao'].iloc[0] - df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['R_sys'].iloc[0] - df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['SV'].iloc[0] - df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['P_sys'].iloc[0] - df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_mod_tono_post_open[df_mod_tono_post_open['partid'] == elt]['P_dia'].iloc[0] - df_mod_tono_mid_open[df_mod_tono_mid_open['partid'] == elt]['P_dia'].iloc[0]
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Append
    df_meas_changes_tono_open = df_meas_changes_tono_open.append(dict(zip(change_col,line_vec_meas)),ignore_index=True)
    df_mod_changes_tono_open = df_mod_changes_tono_open.append(dict(zip(change_col,line_vec_mod)),ignore_index=True)



# Repeat for finger pressure (fp)
df_meas_changes_fp_open = pd.DataFrame(columns=change_col)
df_mod_changes_fp_open = pd.DataFrame(columns=change_col)

for elt in common_ids_fp_open:
    # Meas
    P_sys_6wk = df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['P_sys'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['P_dia'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['SV'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['SV'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['P_sys'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['P_dia'].iloc[0] - df_meas_fp_pre_open[df_meas_fp_pre_open['partid'] == elt]['P_dia'].iloc[0]

    E_max_612wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['SV'].iloc[0] - df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['P_sys'].iloc[0] - df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_meas_fp_post_open[df_meas_fp_post_open['partid'] == elt]['P_dia'].iloc[0] - df_meas_fp_mid_open[df_meas_fp_mid_open['partid'] == elt]['P_dia'].iloc[0]
    
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Mod
    P_sys_6wk = df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['P_sys'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_6wk = df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['P_dia'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['P_dia'].iloc[0]
    SV_6wk = df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['SV'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['SV'].iloc[0]
    R_sys_6wk = df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['R_sys'].iloc[0]
    C_ao_6wk = df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['C_ao'].iloc[0]
    E_max_6wk = df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['E_max'].iloc[0]
    
    E_max_12wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_12wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_12wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['R_sys'].iloc[0]
    SV_12wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['SV'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['SV'].iloc[0]
    P_sys_12wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['P_sys'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_12wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['P_dia'].iloc[0] - df_mod_fp_pre_open[df_mod_fp_pre_open['partid'] == elt]['P_dia'].iloc[0]

    E_max_612wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['E_max'].iloc[0]
    C_ao_612wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['C_ao'].iloc[0]
    R_sys_612wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['R_sys'].iloc[0]
    SV_612wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['SV'].iloc[0] - df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['SV'].iloc[0]
    P_sys_612wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['P_sys'].iloc[0] - df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['P_sys'].iloc[0]
    P_dia_612wk = df_mod_fp_post_open[df_mod_fp_post_open['partid'] == elt]['P_dia'].iloc[0] - df_mod_fp_mid_open[df_mod_fp_mid_open['partid'] == elt]['P_dia'].iloc[0]
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Append
    df_meas_changes_fp_open = df_meas_changes_fp_open.append(dict(zip(change_col,line_vec_meas)),ignore_index=True)
    df_mod_changes_fp_open = df_mod_changes_fp_open.append(dict(zip(change_col,line_vec_mod)),ignore_index=True)


# Compute correlation values between paired subsets of the data for the open-loop model

# 06, 012

Rsys_6_12_C_open = pgn.corr(df_mod_changes_tono_open["R_sys_6wk"], df_mod_changes_tono_open["R_sys_12wk"])
Cao_6_12_C_open = pgn.corr(df_mod_changes_tono_open["C_ao_6wk"], df_mod_changes_tono_open["C_ao_12wk"])
Emax_6_12_C_open = pgn.corr(df_mod_changes_tono_open["E_max_6wk"], df_mod_changes_tono_open["E_max_12wk"])

Rsys_6_12_D_open = pgn.corr(df_mod_changes_fp_open["R_sys_6wk"], df_mod_changes_fp_open["R_sys_12wk"])
Cao_6_12_D_open = pgn.corr(df_mod_changes_fp_open["C_ao_6wk"], df_mod_changes_fp_open["C_ao_12wk"])
Emax_6_12_D_open = pgn.corr(df_mod_changes_fp_open["E_max_6wk"], df_mod_changes_fp_open["E_max_12wk"])

# 6, 6-12

Rsys_6_612_C_open = pgn.corr(df_mod_changes_tono_open["R_sys_6wk"], df_mod_changes_tono_open["R_sys_612wk"])
Cao_6_612_C_open = pgn.corr(df_mod_changes_tono_open["C_ao_6wk"], df_mod_changes_tono_open["C_ao_612wk"])
Emax_6_612_C_open = pgn.corr(df_mod_changes_tono_open["E_max_6wk"], df_mod_changes_tono_open["E_max_612wk"])

Rsys_6_612_D_open = pgn.corr(df_mod_changes_fp_open["R_sys_6wk"], df_mod_changes_fp_open["R_sys_612wk"])
Cao_6_612_D_open = pgn.corr(df_mod_changes_fp_open["C_ao_6wk"], df_mod_changes_fp_open["C_ao_612wk"])
Emax_6_612_D_open = pgn.corr(df_mod_changes_fp_open["E_max_6wk"], df_mod_changes_fp_open["E_max_612wk"])

# 6-12, 12

Rsys_612_12_C_open = pgn.corr(df_mod_changes_tono_open["R_sys_612wk"], df_mod_changes_tono_open["R_sys_12wk"])
Cao_612_12_C_open = pgn.corr(df_mod_changes_tono_open["C_ao_612wk"], df_mod_changes_tono_open["C_ao_12wk"])
Emax_612_12_C_open = pgn.corr(df_mod_changes_tono_open["E_max_612wk"], df_mod_changes_tono_open["E_max_12wk"])

Rsys_612_12_D_open = pgn.corr(df_mod_changes_fp_open["R_sys_612wk"], df_mod_changes_fp_open["R_sys_12wk"])
Cao_612_12_D_open = pgn.corr(df_mod_changes_fp_open["C_ao_612wk"], df_mod_changes_fp_open["C_ao_12wk"])
Emax_612_12_D_open = pgn.corr(df_mod_changes_fp_open["E_max_612wk"], df_mod_changes_fp_open["E_max_12wk"])


idxes = np.array([1,2,3])

import matplotlib as mpl
import matplotlib.patches as mpatches

########################################################
# Heatmaps
########################################################
########################################################
# Barplots R_sys
idxes = idxes-0.5
fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (6,12), dpi=300)
mpl.rcParams.update({'font.size': 14})
ax1mat = [[Rsys_6_12_C["r"].iloc[0],Rsys_6_612_C["r"].iloc[0], Rsys_612_12_C["r"].iloc[0]],
          [Rsys_6_12_D["r"].iloc[0],Rsys_6_612_D["r"].iloc[0], Rsys_612_12_D["r"].iloc[0]],
          [Rsys_6_12_C_open["r"].iloc[0],Rsys_6_612_C_open["r"].iloc[0], Rsys_612_12_C_open["r"].iloc[0]],
          [Rsys_6_12_D_open["r"].iloc[0],Rsys_6_612_D_open["r"].iloc[0], Rsys_612_12_D_open["r"].iloc[0]],
          [Rsys_6_12_me_C["r"].iloc[0],Rsys_6_612_me_C["r"].iloc[0], Rsys_612_12_me_C["r"].iloc[0]],
          [Rsys_6_12_me_D["r"].iloc[0],Rsys_6_612_me_D["r"].iloc[0], Rsys_612_12_me_D["r"].iloc[0]]]

ylabs = ['CL-C','CL-F','OL-C','OL-F','Conv.-C','Conv.-F']
xlabs = ["1-2,1-3","1-2,2-3","2-3,1-3"]
g1 = sb.heatmap(ax1mat, ax = ax1, annot=True, yticklabels=ylabs, xticklabels=xlabs, vmin=-1., vmax=1., fmt = '.2f')

ax1.set_ylabel('$r$, [-]', fontsize='18')     
ax1.set_title("Correlations of $R_{\mathrm{sys}}$ changes over varying\nperiods")


blank_patch = mpatches.Patch(facecolor='white', edgecolor='black',label='CL')
dotted_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='oo', label='OL')
striped_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Direct')

red_patch = mpatches.Patch(color='red', label='C')
blue_patch = mpatches.Patch(color='blue', label='F')

# Barplots C_ao
ax2mat = [[Cao_6_12_C["r"].iloc[0],Cao_6_612_C["r"].iloc[0], Cao_612_12_C["r"].iloc[0]],
          [Cao_6_12_D["r"].iloc[0],Cao_6_612_D["r"].iloc[0], Cao_612_12_D["r"].iloc[0]],
          [Cao_6_12_C_open["r"].iloc[0],Cao_6_612_C_open["r"].iloc[0], Cao_612_12_C_open["r"].iloc[0]],
          [Cao_6_12_D_open["r"].iloc[0],Cao_6_612_D_open["r"].iloc[0], Cao_612_12_D_open["r"].iloc[0]],
          [Cao_6_12_me_C["r"].iloc[0],Cao_6_612_me_C["r"].iloc[0], Cao_612_12_me_C["r"].iloc[0]],
          [Cao_6_12_me_D["r"].iloc[0],Cao_6_612_me_D["r"].iloc[0], Cao_612_12_me_D["r"].iloc[0]]]

g2 = sb.heatmap(ax2mat, ax = ax2, annot=True, yticklabels=ylabs, xticklabels=xlabs, vmin=-1., vmax=1., fmt = '.2f')
ax2.set_ylabel('$r$, [-]', fontsize='18')    
ax2.set_title("Correlations of $C_{\mathrm{ao}}$ changes over varying\nperiods")


# Barplots E_max
ax3mat = [[Emax_6_12_C["r"].iloc[0], Emax_6_612_C["r"].iloc[0], Emax_612_12_C["r"].iloc[0]], 
          [Emax_6_12_D["r"].iloc[0], Emax_6_612_D["r"].iloc[0], Emax_612_12_D["r"].iloc[0]], 
          [Emax_6_12_C_open["r"].iloc[0],Emax_6_612_C_open["r"].iloc[0], Emax_612_12_C_open["r"].iloc[0]], 
          [Emax_6_12_D_open["r"].iloc[0],Emax_6_612_D_open["r"].iloc[0], Emax_612_12_D_open["r"].iloc[0]],
          [Emax_6_12_me_C["r"].iloc[0],Emax_6_612_me_C["r"].iloc[0], Emax_612_12_me_C["r"].iloc[0]],
          [Emax_6_12_me_D["r"].iloc[0],Emax_6_612_me_D["r"].iloc[0], Emax_612_12_me_D["r"].iloc[0]]]



g3 = sb.heatmap(ax3mat, ax = ax3, annot=True, yticklabels=ylabs, xticklabels=xlabs, vmin=-1., vmax=1., fmt = '.2f')
ax3.set_xlabel('Compared periods, [Test days]', fontsize='18')
ax3.set_ylabel('$r$, [-]', fontsize='18')    
ax3.set_title("Correlations of $E_{\mathrm{max}}$ changes over varying\nperiods")


plt.subplots_adjust(hspace=0.5)
plt.savefig("CorrelationChanges/RsysCaoEmaxIntraCorr_Heatmap_BSA.svg", bbox_inches='tight') 
plt.savefig("CorrelationChanges/RsysCaoEmaxIntraCorr_Heatmap_BSA.pdf", bbox_inches='tight')  
plt.show() 
plt.close() 


