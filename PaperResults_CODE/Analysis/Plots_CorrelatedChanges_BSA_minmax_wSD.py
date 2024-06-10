import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pingouin as pgn


# Load Carotid Data

df_meas_tono = pd.read_pickle('Outputs_measurements_tono_BSA_WF.pkl')
df_mod_tono = pd.read_pickle('Outputs_models_tono_BSA_WF.pkl')

# Load Finger Data

df_meas_FP = pd.read_pickle('Outputs_measurements_FP_BSA_WF.pkl')
df_mod_FP = pd.read_pickle('Outputs_models_FP_BSA_WF.pkl')

# Check for invalid E_max measurement among finger pressure results
for idx, row in df_meas_FP.iterrows():
    if (row["E_max"] > 20.) or (row["E_max"] < 0.):
        print("Invalid E_max at (ID,idx):",row["id"],idx)

print("Search complete")


# Split Dataset by measurement day

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


### LOOP THROUGH FINGER MEASUREMENTS TO COMPUTE CHANGES
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
    
    #['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, R_sys_6wk, C_ao_6wk, E_max_6wk, P_sys_12wk, P_dia_12wk, SV_12wk, R_sys_12wk, C_ao_12wk, E_max_12wk, P_sys_612wk, P_dia_612wk, SV_612wk, R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    # Append
    df_meas_changes_fp = df_meas_changes_fp.append(dict(zip(change_col,line_vec_meas)),ignore_index=True)
    df_mod_changes_fp = df_mod_changes_fp.append(dict(zip(change_col,line_vec_mod)),ignore_index=True)



# Compute correlation values between paired subsets of the data 

# 6, 12
Rsys_6_12_me_C = pgn.corr(df_meas_changes_tono["R_sys_6wk"], df_meas_changes_tono["R_sys_12wk"])
#_,r_6_12_Rsys_me_C,CI_6_12_Rsys_me_C, p_6_12_Rsys_me_C,_,_ = pgn.corr(df_meas_changes_tono["R_sys_6wk"], df_meas_changes_tono["R_sys_12wk"])
Cao_6_12_me_C = pgn.corr(df_meas_changes_tono["C_ao_6wk"], df_meas_changes_tono["C_ao_12wk"])
#_,r_6_12_Cao_me_C,CI_6_12_Cao_me_C, p_6_12_Cao_me_C,_,_ = pgn.corr(df_meas_changes_tono["C_ao_6wk"], df_meas_changes_tono["C_ao_12wk"])
Emax_6_12_me_C = pgn.corr(df_meas_changes_tono["E_max_6wk"], df_meas_changes_tono["E_max_12wk"])
#_,r_6_12_Emax_me_C,CI_6_12_Emax_me_C, p_6_12_Emax_me_C,_,_ = pgn.corr(df_meas_changes_tono["E_max_6wk"], df_meas_changes_tono["E_max_12wk"])

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





# Compute correlations between model and measurement sourced changes

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


# Compute correlations for all measurements independent of period

mAll_Rsys_C = pgn.corr(np.append(np.append(df_meas_changes_tono["R_sys_6wk"].values, df_meas_changes_tono["R_sys_12wk"].values), df_meas_changes_tono["R_sys_612wk"].values), np.append(np.append(df_mod_changes_tono["R_sys_6wk"].values, df_mod_changes_tono["R_sys_12wk"].values), df_mod_changes_tono["R_sys_612wk"].values))
mAll_Cao_C = pgn.corr(np.append(np.append(df_meas_changes_tono["C_ao_6wk"].values, df_meas_changes_tono["C_ao_12wk"].values), df_meas_changes_tono["C_ao_612wk"].values), np.append(np.append(df_mod_changes_tono["C_ao_6wk"].values, df_mod_changes_tono["C_ao_12wk"].values), df_mod_changes_tono["C_ao_612wk"].values))
mAll_Emax_C = pgn.corr(np.append(np.append(df_meas_changes_tono["E_max_6wk"].values, df_meas_changes_tono["E_max_12wk"].values), df_meas_changes_tono["E_max_612wk"].values), np.append(np.append(df_mod_changes_tono["E_max_6wk"].values, df_mod_changes_tono["E_max_12wk"].values), df_mod_changes_tono["E_max_612wk"].values))

mAll_Rsys_D = pgn.corr(np.append(np.append(df_meas_changes_fp["R_sys_6wk"].values, df_meas_changes_fp["R_sys_12wk"].values), df_meas_changes_fp["R_sys_612wk"].values), np.append(np.append(df_mod_changes_fp["R_sys_6wk"].values, df_mod_changes_fp["R_sys_12wk"].values), df_mod_changes_fp["R_sys_612wk"].values))
mAll_Cao_D = pgn.corr(np.append(np.append(df_meas_changes_fp["C_ao_6wk"].values, df_meas_changes_fp["C_ao_12wk"].values), df_meas_changes_fp["C_ao_612wk"].values), np.append(np.append(df_mod_changes_fp["C_ao_6wk"].values, df_mod_changes_fp["C_ao_12wk"].values), df_mod_changes_fp["C_ao_612wk"].values))
mAll_Emax_D = pgn.corr(np.append(np.append(df_meas_changes_fp["E_max_6wk"].values, df_meas_changes_fp["E_max_12wk"].values), df_meas_changes_fp["E_max_612wk"].values), np.append(np.append(df_mod_changes_fp["E_max_6wk"].values, df_mod_changes_fp["E_max_12wk"].values), df_mod_changes_fp["E_max_612wk"].values))


# Construct tables of correlations
df_out_weeks = pd.DataFrame()

df_out_weeks["Par"] = ["R_sys", "C_ao", "E_max", "R_sys", "C_ao", "E_max", "R_sys", "C_ao", "E_max", "R_sys", "C_ao", "E_max"]
df_out_weeks["WaveForm"] = ["Carotid", "Carotid", "Carotid", "Digital", "Digital", "Digital","Carotid", "Carotid", "Carotid", "Digital", "Digital", "Digital"]
df_out_weeks["Process"] = ["Model", "Model", "Model", "Model", "Model", "Model","Measurement", "Measurement", "Measurement", "Measurement", "Measurement", "Measurement"]
df_out_weeks["r_6_12"] = [Rsys_6_12_C["r"].iloc[0],Cao_6_12_C["r"].iloc[0],Emax_6_12_C["r"].iloc[0],
                          Rsys_6_12_D["r"].iloc[0],Cao_6_12_D["r"].iloc[0],Emax_6_12_D["r"].iloc[0],
                          Rsys_6_12_me_C["r"].iloc[0],Cao_6_12_me_C["r"].iloc[0],Emax_6_12_me_C["r"].iloc[0],
                          Rsys_6_12_me_D["r"].iloc[0],Cao_6_12_me_D["r"].iloc[0],Emax_6_12_me_D["r"].iloc[0]]
df_out_weeks["r_6_612"] = [Rsys_6_612_C["r"].iloc[0],Cao_6_612_C["r"].iloc[0],Emax_6_612_C["r"].iloc[0],
                           Rsys_6_612_D["r"].iloc[0],Cao_6_612_D["r"].iloc[0],Emax_6_612_D["r"].iloc[0],
                           Rsys_6_612_me_C["r"].iloc[0],Cao_6_612_me_C["r"].iloc[0],Emax_6_612_me_C["r"].iloc[0],
                           Rsys_6_612_me_D["r"].iloc[0],Cao_6_612_me_D["r"].iloc[0],Emax_6_612_me_D["r"].iloc[0]]
df_out_weeks["r_612_12"] = [Rsys_612_12_C["r"].iloc[0],Cao_612_12_C["r"].iloc[0],Emax_612_12_C["r"].iloc[0],
                            Rsys_612_12_D["r"].iloc[0],Cao_612_12_D["r"].iloc[0],Emax_612_12_D["r"].iloc[0],
                            Rsys_612_12_me_C["r"].iloc[0],Cao_612_12_me_C["r"].iloc[0],Emax_612_12_me_C["r"].iloc[0],
                            Rsys_612_12_me_D["r"].iloc[0],Cao_612_12_me_D["r"].iloc[0],Emax_612_12_me_D["r"].iloc[0]]

df_out_weeks["p_6_12"] = [Rsys_6_12_C["p-val"].iloc[0],Cao_6_12_C["p-val"].iloc[0],Emax_6_12_C["p-val"].iloc[0],
                          Rsys_6_12_D["p-val"].iloc[0],Cao_6_12_D["p-val"].iloc[0],Emax_6_12_D["p-val"].iloc[0],
                          Rsys_6_12_me_C["p-val"].iloc[0],Cao_6_12_me_C["p-val"].iloc[0],Emax_6_12_me_C["p-val"].iloc[0],
                          Rsys_6_12_me_D["p-val"].iloc[0],Cao_6_12_me_D["p-val"].iloc[0],Emax_6_12_me_D["p-val"].iloc[0]]
df_out_weeks["p_6_612"] = [Rsys_6_612_C["p-val"].iloc[0],Cao_6_612_C["p-val"].iloc[0],Emax_6_612_C["p-val"].iloc[0],
                           Rsys_6_612_D["p-val"].iloc[0],Cao_6_612_D["p-val"].iloc[0],Emax_6_612_D["p-val"].iloc[0],
                           Rsys_6_612_me_C["p-val"].iloc[0],Cao_6_612_me_C["p-val"].iloc[0],Emax_6_612_me_C["p-val"].iloc[0],
                           Rsys_6_612_me_D["p-val"].iloc[0],Cao_6_612_me_D["p-val"].iloc[0],Emax_6_612_me_D["p-val"].iloc[0]]
df_out_weeks["p_612_12"] = [Rsys_612_12_C["p-val"].iloc[0],Cao_612_12_C["p-val"].iloc[0],Emax_612_12_C["p-val"].iloc[0],
                            Rsys_612_12_D["p-val"].iloc[0],Cao_612_12_D["p-val"].iloc[0],Emax_612_12_D["p-val"].iloc[0],
                            Rsys_612_12_me_C["p-val"].iloc[0],Cao_612_12_me_C["p-val"].iloc[0],Emax_612_12_me_C["p-val"].iloc[0],
                            Rsys_612_12_me_D["p-val"].iloc[0],Cao_612_12_me_D["p-val"].iloc[0],Emax_612_12_me_D["p-val"].iloc[0]]

df_out_weeks["CI_6_12"] = [Rsys_6_12_C["CI95%"].iloc[0],Cao_6_12_C["CI95%"].iloc[0],Emax_6_12_C["CI95%"].iloc[0],
                           Rsys_6_12_D["CI95%"].iloc[0],Cao_6_12_D["CI95%"].iloc[0],Emax_6_12_D["CI95%"].iloc[0],
                           Rsys_6_12_me_C["CI95%"].iloc[0],Cao_6_12_me_C["CI95%"].iloc[0],Emax_6_12_me_C["CI95%"].iloc[0],
                           Rsys_6_12_me_D["CI95%"].iloc[0],Cao_6_12_me_D["CI95%"].iloc[0],Emax_6_12_me_D["CI95%"].iloc[0]]
df_out_weeks["CI_6_612"] = [Rsys_6_612_C["CI95%"].iloc[0],Cao_6_612_C["CI95%"].iloc[0],Emax_6_612_C["CI95%"].iloc[0],
                            Rsys_6_612_D["CI95%"].iloc[0],Cao_6_612_D["CI95%"].iloc[0],Emax_6_612_D["CI95%"].iloc[0],
                            Rsys_6_612_me_C["CI95%"].iloc[0],Cao_6_612_me_C["CI95%"].iloc[0],Emax_6_612_me_C["CI95%"].iloc[0],
                            Rsys_6_612_me_D["CI95%"].iloc[0],Cao_6_612_me_D["CI95%"].iloc[0],Emax_6_612_me_D["CI95%"].iloc[0]]
df_out_weeks["CI_612_12"] = [Rsys_612_12_C["CI95%"].iloc[0],Cao_612_12_C["CI95%"].iloc[0],Emax_612_12_C["CI95%"].iloc[0],
                                Rsys_612_12_D["CI95%"].iloc[0],Cao_612_12_D["CI95%"].iloc[0],Emax_612_12_D["CI95%"].iloc[0],
                                Rsys_612_12_me_C["CI95%"].iloc[0],Cao_612_12_me_C["CI95%"].iloc[0],Emax_612_12_me_C["CI95%"].iloc[0],
                                Rsys_612_12_me_D["CI95%"].iloc[0],Cao_612_12_me_D["CI95%"].iloc[0],Emax_612_12_me_D["CI95%"].iloc[0]]

print(df_out_weeks)

# Construct tables for correlations between model estimated and conventionally estimated changes. 

df_out_modmeas = pd.DataFrame()
df_out_modmeas["Par"] = ["R_sys", "C_ao", "E_max", "R_sys", "C_ao", "E_max"]
df_out_modmeas["WaveForm"] = ["Carotid", "Carotid", "Carotid", "Digital", "Digital", "Digital"]

df_out_modmeas["r_6"] = [m6_Rsys_C["r"].iloc[0], m6_Cao_C["r"].iloc[0], m6_Emax_C["r"].iloc[0], m6_Rsys_D["r"].iloc[0], m6_Cao_D["r"].iloc[0], m6_Emax_D["r"].iloc[0]]
df_out_modmeas["r_12"] = [m12_Rsys_C["r"].iloc[0], m12_Cao_C["r"].iloc[0], m12_Emax_C["r"].iloc[0], m12_Rsys_D["r"].iloc[0], m12_Cao_D["r"].iloc[0], m12_Emax_D["r"].iloc[0]]
df_out_modmeas["r_612"] = [m612_Rsys_C["r"].iloc[0], m612_Cao_C["r"].iloc[0], m612_Emax_C["r"].iloc[0], m612_Rsys_D["r"].iloc[0], m612_Cao_D["r"].iloc[0], m612_Emax_D["r"].iloc[0]]
df_out_modmeas["r_All"] = [mAll_Rsys_C["r"].iloc[0], mAll_Cao_C["r"].iloc[0], mAll_Emax_C["r"].iloc[0], mAll_Rsys_D["r"].iloc[0], mAll_Cao_D["r"].iloc[0], mAll_Emax_D["r"].iloc[0]]

df_out_modmeas["p_6"] = [m6_Rsys_C["p-val"].iloc[0], m6_Cao_C["p-val"].iloc[0], m6_Emax_C["p-val"].iloc[0], m6_Rsys_D["p-val"].iloc[0], m6_Cao_D["p-val"].iloc[0], m6_Emax_D["p-val"].iloc[0]]
df_out_modmeas["p_12"] = [m12_Rsys_C["p-val"].iloc[0], m12_Cao_C["p-val"].iloc[0], m12_Emax_C["p-val"].iloc[0], m12_Rsys_D["p-val"].iloc[0], m12_Cao_D["p-val"].iloc[0], m12_Emax_D["p-val"].iloc[0]]
df_out_modmeas["p_612"] = [m612_Rsys_C["p-val"].iloc[0], m612_Cao_C["p-val"].iloc[0], m612_Emax_C["p-val"].iloc[0], m612_Rsys_D["p-val"].iloc[0], m612_Cao_D["p-val"].iloc[0], m612_Emax_D["p-val"].iloc[0]]
df_out_modmeas["p_All"] = [mAll_Rsys_C["p-val"].iloc[0], mAll_Cao_C["p-val"].iloc[0], mAll_Emax_C["p-val"].iloc[0], mAll_Rsys_D["p-val"].iloc[0], mAll_Cao_D["p-val"].iloc[0], mAll_Emax_D["p-val"].iloc[0]]

df_out_modmeas["CI_6"] = [m6_Rsys_C["CI95%"].iloc[0], m6_Cao_C["CI95%"].iloc[0], m6_Emax_C["CI95%"].iloc[0], m6_Rsys_D["CI95%"].iloc[0], m6_Cao_D["CI95%"].iloc[0], m6_Emax_D["CI95%"].iloc[0]]
df_out_modmeas["CI_12"] = [m12_Rsys_C["CI95%"].iloc[0], m12_Cao_C["CI95%"].iloc[0], m12_Emax_C["CI95%"].iloc[0], m12_Rsys_D["CI95%"].iloc[0], m12_Cao_D["CI95%"].iloc[0], m12_Emax_D["CI95%"].iloc[0]]
df_out_modmeas["CI_612"] = [m612_Rsys_C["CI95%"].iloc[0], m612_Cao_C["CI95%"].iloc[0], m612_Emax_C["CI95%"].iloc[0], m612_Rsys_D["CI95%"].iloc[0], m612_Cao_D["CI95%"].iloc[0], m612_Emax_D["CI95%"].iloc[0]]
df_out_modmeas["CI_All"] = [mAll_Rsys_C["CI95%"].iloc[0], mAll_Cao_C["CI95%"].iloc[0], mAll_Emax_C["CI95%"].iloc[0], mAll_Rsys_D["CI95%"].iloc[0], mAll_Cao_D["CI95%"].iloc[0], mAll_Emax_D["CI95%"].iloc[0]]

print(df_out_modmeas)

df_out_weeks.to_csv('CorrelationChanges/TableChangesBothWFAllWeeks_BSA.csv')
df_out_weeks.to_excel('CorrelationChanges/TableChangesBothWFAllWeeks_BSA.xlsx')

df_out_modmeas.to_csv('CorrelationChanges/TableModMeasBothWFAllWeeks_BSA.csv')
df_out_modmeas.to_excel('CorrelationChanges/TableModMeasBothWFAllWeeks_BSA.xlsx')



# Compute some descriptive statistics about changes 
# Maximal changes all week changes

df_max = pd.DataFrame()

df_max["Par"] = ["R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max"] 
df_max["WaveForm"] = ["Carotid", "Carotid", "Carotid","Digital","Digital","Digital","Carotid", "Carotid", "Carotid","Digital","Digital","Digital"]
df_max["Process"] = ["Model", "Model", "Model", "Model", "Model", "Model","Measurement", "Measurement", "Measurement", "Measurement", "Measurement", "Measurement"]

df_max["Maximal change"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().max().max(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().max().max(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().max().max(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().max().max(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().max().max(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().max().max(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().max().max(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().max().max(), df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().max().max(),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().max().max(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().max().max(), df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().max().max()]

df_max["Minimal change"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().min().min(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().min().min(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().min().min(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().min().min(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().min().min(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().min().min(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().min().min(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().min().min(), df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().min().min(),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().min().min(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().min().min(), df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().min().min()]

df_max["Maximal signed change"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].max().max(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].max().max(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].max().max(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].max().max(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].max().max(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].max().max(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].max().max(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].max().max(), df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].max().max(),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].max().max(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].max().max(), df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].max().max()]

df_max["Minimal signed change"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].min().min(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].min().min(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].min().min(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].min().min(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].min().min(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].min().min(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].min().min(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].min().min(), df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].min().min(),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].min().min(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].min().min(), df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].min().min()]


print(len(df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten()), df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().mean(),  (df_mod_changes_tono["R_sys_12wk"].sum() + df_mod_changes_tono["R_sys_6wk"].sum() + df_mod_changes_tono["R_sys_612wk"].sum())/(3*len(df_mod_changes_tono.index)) )

df_max["Average change"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().mean(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().mean(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten().mean(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().mean(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().mean(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten().mean(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().mean(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().mean(), np.nanmean(df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten()),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().mean(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().mean(), np.nanmean(df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten())]

df_max["Average change SD"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().std(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().std(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten().std(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().std(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().std(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten().std(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().std(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().std(), np.nanstd(df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten()),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].values.flatten().std(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].values.flatten().std(), np.nanstd(df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].values.flatten())]

df_max["Average absolute change"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().mean(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().mean(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten().mean(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().mean(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().mean(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten().mean(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().mean(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().mean(), np.nanmean(df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten()),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().mean(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().mean(), np.nanmean(df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten())]

df_max["Average absolute change SD"] = [df_mod_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().std(), df_mod_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().std(), df_mod_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten().std(),
df_mod_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().std(), df_mod_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().std(), df_mod_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten().std(),
df_meas_changes_tono[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().std(), df_meas_changes_tono[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().std(), np.nanstd(df_meas_changes_tono[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten()),
df_meas_changes_fp[["R_sys_12wk","R_sys_6wk","R_sys_612wk"]].abs().values.flatten().std(), df_meas_changes_fp[["C_ao_12wk","C_ao_6wk","C_ao_612wk"]].abs().values.flatten().std(), np.nanstd(df_meas_changes_fp[["E_max_12wk","E_max_6wk","E_max_612wk"]].abs().values.flatten())]

print(df_max)

# Compute the same statistics but only for the change from measurement day 1 to measurement day 3
# Maximal changes after 12 weeks

df_max_12wk = pd.DataFrame()#columns = ["Maximal change", "Minimal change", "Average change", "Average absolute change"])
df_max_12wk["Par"] = ["R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max"] 
df_max_12wk["WaveForm"] = ["Carotid", "Carotid", "Carotid","Digital","Digital","Digital","Carotid", "Carotid", "Carotid","Digital","Digital","Digital"]
df_max_12wk["Process"] = ["Model", "Model", "Model", "Model", "Model", "Model","Measurement", "Measurement", "Measurement", "Measurement", "Measurement", "Measurement"]

df_max_12wk["Maximal change"] = [df_mod_changes_tono[["R_sys_12wk"]].abs().max().max(), df_mod_changes_tono[["C_ao_12wk"]].abs().max().max(), df_mod_changes_tono[["E_max_12wk"]].abs().max().max(),
df_mod_changes_fp[["R_sys_12wk"]].abs().max().max(), df_mod_changes_fp[["C_ao_12wk"]].abs().max().max(), df_mod_changes_fp[["E_max_12wk"]].abs().max().max(),
df_meas_changes_tono[["R_sys_12wk"]].abs().max().max(), df_meas_changes_tono[["C_ao_12wk"]].abs().max().max(), df_meas_changes_tono[["E_max_12wk"]].abs().max().max(),
df_meas_changes_fp[["R_sys_12wk"]].abs().max().max(), df_meas_changes_fp[["C_ao_12wk"]].abs().max().max(), df_meas_changes_fp[["E_max_12wk"]].abs().max().max()]

df_max_12wk["Minimal change"] = [df_mod_changes_tono[["R_sys_12wk"]].abs().min().min(), df_mod_changes_tono[["C_ao_12wk"]].abs().min().min(), df_mod_changes_tono[["E_max_12wk"]].abs().min().min(),
df_mod_changes_fp[["R_sys_12wk"]].abs().min().min(), df_mod_changes_fp[["C_ao_12wk"]].abs().min().min(), df_mod_changes_fp[["E_max_12wk"]].abs().min().min(),
df_meas_changes_tono[["R_sys_12wk"]].abs().min().min(), df_meas_changes_tono[["C_ao_12wk"]].abs().min().min(), df_meas_changes_tono[["E_max_12wk"]].abs().min().min(),
df_meas_changes_fp[["R_sys_12wk"]].abs().min().min(), df_meas_changes_fp[["C_ao_12wk"]].abs().min().min(), df_meas_changes_fp[["E_max_12wk"]].abs().min().min()]

df_max_12wk["Maximal signed change"] = [df_mod_changes_tono[["R_sys_12wk"]].max().max(), df_mod_changes_tono[["C_ao_12wk"]].max().max(), df_mod_changes_tono[["E_max_12wk"]].max().max(),
df_mod_changes_fp[["R_sys_12wk"]].max().max(), df_mod_changes_fp[["C_ao_12wk"]].max().max(), df_mod_changes_fp[["E_max_12wk"]].max().max(),
df_meas_changes_tono[["R_sys_12wk"]].max().max(), df_meas_changes_tono[["C_ao_12wk"]].max().max(), df_meas_changes_tono[["E_max_12wk"]].max().max(),
df_meas_changes_fp[["R_sys_12wk"]].max().max(), df_meas_changes_fp[["C_ao_12wk"]].max().max(), df_meas_changes_fp[["E_max_12wk"]].max().max()]

df_max_12wk["Minimal signed change"] = [df_mod_changes_tono[["R_sys_12wk"]].min().min(), df_mod_changes_tono[["C_ao_12wk"]].min().min(), df_mod_changes_tono[["E_max_12wk"]].min().min(),
df_mod_changes_fp[["R_sys_12wk"]].min().min(), df_mod_changes_fp[["C_ao_12wk"]].min().min(), df_mod_changes_fp[["E_max_12wk"]].min().min(),
df_meas_changes_tono[["R_sys_12wk"]].min().min(), df_meas_changes_tono[["C_ao_12wk"]].min().min(), df_meas_changes_tono[["E_max_12wk"]].min().min(),
df_meas_changes_fp[["R_sys_12wk"]].min().min(), df_meas_changes_fp[["C_ao_12wk"]].min().min(), df_meas_changes_fp[["E_max_12wk"]].min().min()]


df_max_12wk["Average change"] = [df_mod_changes_tono[["R_sys_12wk"]].mean().values[0], df_mod_changes_tono[["C_ao_12wk"]].mean().values[0], df_mod_changes_tono[["E_max_12wk"]].mean().values[0],
df_mod_changes_fp[["R_sys_12wk"]].mean().values[0], df_mod_changes_fp[["C_ao_12wk"]].mean().values[0], df_mod_changes_fp[["E_max_12wk"]].mean().values[0],
df_meas_changes_tono[["R_sys_12wk"]].mean().values[0], df_meas_changes_tono[["C_ao_12wk"]].mean().values[0], np.nanmean(df_meas_changes_tono[["E_max_12wk"]]),
df_meas_changes_fp[["R_sys_12wk"]].mean().values[0], df_meas_changes_fp[["C_ao_12wk"]].mean().values[0], np.nanmean(df_meas_changes_fp[["E_max_12wk"]])]

df_max_12wk["Average absolute change"] = [df_mod_changes_tono[["R_sys_12wk"]].abs().values.flatten().mean(), df_mod_changes_tono[["C_ao_12wk"]].abs().values.flatten().mean(), df_mod_changes_tono[["E_max_12wk"]].abs().values.flatten().mean(),
df_mod_changes_fp[["R_sys_12wk"]].abs().values.flatten().mean(), df_mod_changes_fp[["C_ao_12wk"]].abs().values.flatten().mean(), df_mod_changes_fp[["E_max_12wk"]].abs().values.flatten().mean(),
df_meas_changes_tono[["R_sys_12wk"]].abs().values.flatten().mean(), df_meas_changes_tono[["C_ao_12wk"]].abs().values.flatten().mean(), np.nanmean(df_meas_changes_tono[["E_max_12wk"]].abs()),
df_meas_changes_fp[["R_sys_12wk"]].abs().values.flatten().mean(), df_meas_changes_fp[["C_ao_12wk"]].abs().values.flatten().mean(), np.nanmean(df_meas_changes_fp[["E_max_12wk"]].abs())]

print(df_max_12wk)

# Compute the same statistics but only for the change from measurement day 1 to measurement day 2
# Maximal changes after 6 weeks

df_max_6wk = pd.DataFrame()

df_max_6wk["Par"] = ["R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max","R_sys", "C_ao", "E_max"] 
df_max_6wk["WaveForm"] = ["Carotid", "Carotid", "Carotid","Digital","Digital","Digital","Carotid", "Carotid", "Carotid","Digital","Digital","Digital"]
df_max_6wk["Process"] = ["Model", "Model", "Model", "Model", "Model", "Model","Measurement", "Measurement", "Measurement", "Measurement", "Measurement", "Measurement"]

df_max_6wk["Maximal change"] = [df_mod_changes_tono[["R_sys_6wk"]].abs().max().max(), df_mod_changes_tono[["C_ao_6wk"]].abs().max().max(), df_mod_changes_tono[["E_max_6wk"]].abs().max().max(),
df_mod_changes_fp[["R_sys_6wk"]].abs().max().max(), df_mod_changes_fp[["C_ao_6wk"]].abs().max().max(), df_mod_changes_fp[["E_max_6wk"]].abs().max().max(),
df_meas_changes_tono[["R_sys_6wk"]].abs().max().max(), df_meas_changes_tono[["C_ao_6wk"]].abs().max().max(), df_meas_changes_tono[["E_max_6wk"]].abs().max().max(),
df_meas_changes_fp[["R_sys_6wk"]].abs().max().max(), df_meas_changes_fp[["C_ao_6wk"]].abs().max().max(), df_meas_changes_fp[["E_max_6wk"]].abs().max().max()]

df_max_6wk["Minimal change"] = [df_mod_changes_tono[["R_sys_6wk"]].abs().min().min(), df_mod_changes_tono[["C_ao_6wk"]].abs().min().min(), df_mod_changes_tono[["E_max_6wk"]].abs().min().min(),
df_mod_changes_fp[["R_sys_6wk"]].abs().min().min(), df_mod_changes_fp[["C_ao_6wk"]].abs().min().min(), df_mod_changes_fp[["E_max_6wk"]].abs().min().min(),
df_meas_changes_tono[["R_sys_6wk"]].abs().min().min(), df_meas_changes_tono[["C_ao_6wk"]].abs().min().min(), df_meas_changes_tono[["E_max_6wk"]].abs().min().min(),
df_meas_changes_fp[["R_sys_6wk"]].abs().min().min(), df_meas_changes_fp[["C_ao_6wk"]].abs().min().min(), df_meas_changes_fp[["E_max_6wk"]].abs().min().min()]

df_max_6wk["Maximal signed change"] = [df_mod_changes_tono[["R_sys_6wk"]].max().max(), df_mod_changes_tono[["C_ao_6wk"]].max().max(), df_mod_changes_tono[["E_max_6wk"]].max().max(),
df_mod_changes_fp[["R_sys_6wk"]].max().max(), df_mod_changes_fp[["C_ao_6wk"]].max().max(), df_mod_changes_fp[["E_max_6wk"]].max().max(),
df_meas_changes_tono[["R_sys_6wk"]].max().max(), df_meas_changes_tono[["C_ao_6wk"]].max().max(), df_meas_changes_tono[["E_max_6wk"]].max().max(),
df_meas_changes_fp[["R_sys_6wk"]].max().max(), df_meas_changes_fp[["C_ao_6wk"]].max().max(), df_meas_changes_fp[["E_max_6wk"]].max().max()]

df_max_6wk["Minimal signed change"] = [df_mod_changes_tono[["R_sys_6wk"]].min().min(), df_mod_changes_tono[["C_ao_6wk"]].min().min(), df_mod_changes_tono[["E_max_6wk"]].min().min(),
df_mod_changes_fp[["R_sys_6wk"]].min().min(), df_mod_changes_fp[["C_ao_6wk"]].min().min(), df_mod_changes_fp[["E_max_6wk"]].min().min(),
df_meas_changes_tono[["R_sys_6wk"]].min().min(), df_meas_changes_tono[["C_ao_6wk"]].min().min(), df_meas_changes_tono[["E_max_6wk"]].min().min(),
df_meas_changes_fp[["R_sys_6wk"]].min().min(), df_meas_changes_fp[["C_ao_6wk"]].min().min(), df_meas_changes_fp[["E_max_6wk"]].min().min()]


df_max_6wk["Average change"] = [df_mod_changes_tono[["R_sys_6wk"]].mean().values[0], df_mod_changes_tono[["C_ao_6wk"]].mean().values[0], df_mod_changes_tono[["E_max_6wk"]].mean().values[0],
df_mod_changes_fp[["R_sys_6wk"]].mean().values[0], df_mod_changes_fp[["C_ao_6wk"]].mean().values[0], df_mod_changes_fp[["E_max_6wk"]].mean().values[0],
df_meas_changes_tono[["R_sys_6wk"]].mean().values[0], df_meas_changes_tono[["C_ao_6wk"]].mean().values[0], np.nanmean(df_meas_changes_tono[["E_max_6wk"]]),
df_meas_changes_fp[["R_sys_6wk"]].mean().values[0], df_meas_changes_fp[["C_ao_6wk"]].mean().values[0], np.nanmean(df_meas_changes_fp[["E_max_6wk"]])]

df_max_6wk["Average absolute change"] = [df_mod_changes_tono[["R_sys_6wk"]].abs().values.flatten().mean(), df_mod_changes_tono[["C_ao_6wk"]].abs().values.flatten().mean(), df_mod_changes_tono[["E_max_6wk"]].abs().values.flatten().mean(),
df_mod_changes_fp[["R_sys_6wk"]].abs().values.flatten().mean(), df_mod_changes_fp[["C_ao_6wk"]].abs().values.flatten().mean(), df_mod_changes_fp[["E_max_6wk"]].abs().values.flatten().mean(),
df_meas_changes_tono[["R_sys_6wk"]].abs().values.flatten().mean(), df_meas_changes_tono[["C_ao_6wk"]].abs().values.flatten().mean(), np.nanmean(df_meas_changes_tono[["E_max_6wk"]].abs()),
df_meas_changes_fp[["R_sys_6wk"]].abs().values.flatten().mean(), df_meas_changes_fp[["C_ao_6wk"]].abs().values.flatten().mean(), np.nanmean(df_meas_changes_fp[["E_max_6wk"]].abs())]

print(df_max_6wk)

df_max.to_csv('CorrelationChanges/TableMinMaxChangesBothWFAllWeeks_BSA.csv')
df_max.to_excel('CorrelationChanges/TableMinMaxChangesBothWFAllWeeks_BSA.xlsx')

df_max_6wk.to_csv('CorrelationChanges/TableMinMaxModMeasBothWF6Weeks_BSA.csv')
df_max_6wk.to_excel('CorrelationChanges/TableMinMaxModMeasBothWF6Weeks_BSA.xlsx')

df_max_12wk.to_csv('CorrelationChanges/TableMinMaxModMeasBothWF12Weeks_BSA.csv')
df_max_12wk.to_excel('CorrelationChanges/TableMinMaxModMeasBothWF12Weeks_BSA.xlsx')

