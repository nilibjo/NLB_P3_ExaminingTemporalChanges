import pandas as pd
import numpy as np
import os


# Load Carotid Pressure Data

df_meas_tono = pd.read_pickle('Outputs_measurements_tono_BSA_WF.pkl')
df_mod_tono = pd.read_pickle('Outputs_models_tono_BSA_WF.pkl')

# Load Finger Pressure Data

df_meas_FP = pd.read_pickle('Outputs_measurements_FP_BSA_WF.pkl')
df_mod_FP = pd.read_pickle('Outputs_models_FP_BSA_WF.pkl')

# Load Post CPET Finger Pressure Data

df_meas_FP_CPET = pd.read_pickle('Outputs_CPET_measurements_FP_BSA_CPET.pkl')
df_mod_FP_CPET = pd.read_pickle('Outputs_CPET_models_FP_BSA_CPET.pkl')


# Check for invalid E_max in finger pressure measurement derived data
remove_idx = []
for idx, row in df_meas_FP.iterrows():
    if (row["E_max"] > 20.) or (row["E_max"] < 0.):
        print("Invalid E_max at (ID,idx):",row["id"],idx)
        remove_idx.append(idx)


print("Search for invalid E_max values complete")


# Split dataset by measurement day

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

# Set up data structures for storing the computed changes
change_col = ['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk", "R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET", "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET", "R_sys_12wk_CPET", "C_ao_12wk_CPET", "E_max_12wk_CPET"]

plot_col=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"]
df_meas_plot_C = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"])
df_meas_plot_D = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"])
df_mod_plot_C = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"])
df_mod_plot_D = pd.DataFrame(columns=["id", "R1", "R2", "R3", "C1", "C2", "C3", "E1", "E2", "E3"])

df_meas_changes_tono = pd.DataFrame(columns=change_col)
df_mod_changes_tono = pd.DataFrame(columns=change_col)

# Split dataset by measurement day for CPET measurements
df_meas_fp_CPET_pre = df_meas_FP_CPET[df_meas_FP_CPET['test_day'] == 'Pre-test day 3']
df_meas_fp_CPET_post = df_meas_FP_CPET[df_meas_FP_CPET['test_day'] == 'Post-test day 3'] 

df_mod_fp_CPET_pre = df_mod_FP_CPET[df_mod_FP_CPET['test_day'] == 'Pre-test day 3']
df_mod_fp_CPET_post = df_mod_FP_CPET[df_mod_FP_CPET['test_day'] == 'Post-test day 3'] 

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
    
    CPET_R_sys_pre = np.nan
    CPET_E_max_pre = np.nan
    CPET_C_ao_pre = np.nan
    
    CPET_R_sys_post = np.nan
    CPET_C_ao_post = np.nan
    CPET_E_max_post = np.nan
    CPET_R_sys_12wk = np.nan
    CPET_C_ao_12wk = np.nan
    CPET_E_max_12wk = np.nan
    
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                          R_sys_6wk, C_ao_6wk, E_max_6wk, 
                          P_sys_12wk, P_dia_12wk, SV_12wk, 
                          R_sys_12wk, C_ao_12wk, E_max_12wk, 
                          P_sys_612wk, P_dia_612wk, SV_612wk, 
                          R_sys_612wk, C_ao_612wk, E_max_612wk, 
                          CPET_R_sys_pre, CPET_C_ao_pre, CPET_E_max_pre, 
                          CPET_R_sys_post, CPET_C_ao_post, CPET_E_max_post, 
                          CPET_R_sys_12wk, CPET_C_ao_12wk, CPET_E_max_12wk]
    
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
    
    CPET_R_sys_pre = np.nan
    CPET_E_max_pre = np.nan
    CPET_C_ao_pre = np.nan

    CPET_R_sys_post = np.nan
    CPET_C_ao_post = np.nan
    CPET_E_max_post = np.nan
    CPET_R_sys_12wk = np.nan
    CPET_C_ao_12wk = np.nan
    CPET_E_max_12wk = np.nan
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                          R_sys_6wk, C_ao_6wk, E_max_6wk, 
                          P_sys_12wk, P_dia_12wk, SV_12wk, 
                          R_sys_12wk, C_ao_12wk, E_max_12wk, 
                          P_sys_612wk, P_dia_612wk, SV_612wk, 
                          R_sys_612wk, C_ao_612wk, E_max_612wk, 
                          CPET_R_sys_pre, CPET_C_ao_pre, CPET_E_max_pre, 
                          CPET_R_sys_post, CPET_C_ao_post, CPET_E_max_post, 
                          CPET_R_sys_12wk, CPET_C_ao_12wk, CPET_E_max_12wk]
    
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
    else:
        CPET_R_sys_pre = np.nan
        CPET_E_max_pre = np.nan
        CPET_C_ao_pre = np.nan

    if elt in CPET_post_ids:
        CPET_R_sys_12wk = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_12wk = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_12wk = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_CPET_pre[df_meas_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0] 
        CPET_R_sys_post = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_meas_fp_post[df_meas_fp_post['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_post = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_meas_fp_post[df_meas_fp_post['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_post = df_meas_fp_CPET_post[df_meas_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_meas_fp_post[df_meas_fp_post['partid'] == elt]['C_ao'].iloc[0] 
    else:
        CPET_R_sys_post = np.nan
        CPET_C_ao_post = np.nan
        CPET_E_max_post = np.nan
        CPET_R_sys_12wk = np.nan
        CPET_C_ao_12wk = np.nan
        CPET_E_max_12wk = np.nan
    
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                          R_sys_6wk, C_ao_6wk, E_max_6wk, 
                          P_sys_12wk, P_dia_12wk, SV_12wk, 
                          R_sys_12wk, C_ao_12wk, E_max_12wk, 
                          P_sys_612wk, P_dia_612wk, SV_612wk, 
                          R_sys_612wk, C_ao_612wk, E_max_612wk, 
                          CPET_R_sys_pre, CPET_C_ao_pre, CPET_E_max_pre, 
                          CPET_R_sys_post, CPET_C_ao_post, CPET_E_max_post, 
                          CPET_R_sys_12wk, CPET_C_ao_12wk, CPET_E_max_12wk]
    
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
    else:
        CPET_R_sys_pre = np.nan
        CPET_E_max_pre = np.nan
        CPET_C_ao_pre = np.nan
    
    if elt in CPET_post_ids:
        CPET_R_sys_12wk = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_12wk = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_12wk = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_CPET_pre[df_mod_fp_CPET_pre['partid'] == elt]['C_ao'].iloc[0] 
        CPET_R_sys_post = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['R_sys'].iloc[0] - df_mod_fp_post[df_mod_fp_post['partid'] == elt]['R_sys'].iloc[0]
        CPET_E_max_post = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['E_max'].iloc[0] - df_mod_fp_post[df_mod_fp_post['partid'] == elt]['E_max'].iloc[0]
        CPET_C_ao_post = df_mod_fp_CPET_post[df_mod_fp_CPET_post['partid'] == elt]['C_ao'].iloc[0] - df_mod_fp_post[df_mod_fp_post['partid'] == elt]['C_ao'].iloc[0] 
    else:
        CPET_R_sys_post = np.nan
        CPET_C_ao_post = np.nan
        CPET_E_max_post = np.nan
        CPET_R_sys_12wk = np.nan
        CPET_C_ao_12wk = np.nan
        CPET_E_max_12wk = np.nan
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                         R_sys_6wk, C_ao_6wk, E_max_6wk, 
                         P_sys_12wk, P_dia_12wk, SV_12wk, 
                         R_sys_12wk, C_ao_12wk, E_max_12wk, 
                         P_sys_612wk, P_dia_612wk, SV_612wk, 
                         R_sys_612wk, C_ao_612wk, E_max_612wk, 
                         CPET_R_sys_pre, CPET_C_ao_pre, CPET_E_max_pre, 
                         CPET_R_sys_post, CPET_C_ao_post, CPET_E_max_post, 
                         CPET_R_sys_12wk, CPET_C_ao_12wk, CPET_E_max_12wk]
    
    
    # Append
    df_meas_changes_fp = df_meas_changes_fp.append(dict(zip(change_col,line_vec_meas)),ignore_index=True)
    df_mod_changes_fp = df_mod_changes_fp.append(dict(zip(change_col,line_vec_mod)),ignore_index=True)
    
    plot_meas_D_vec = [elt, df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['R_sys'].iloc[0], df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['R_sys'].iloc[0], df_meas_fp_post[df_meas_fp_post['partid'] == elt]['R_sys'].iloc[0], df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['C_ao'].iloc[0], df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['C_ao'].iloc[0], df_meas_fp_post[df_meas_fp_post['partid'] == elt]['C_ao'].iloc[0],  df_meas_fp_pre[df_meas_fp_pre['partid'] == elt]['E_max'].iloc[0], df_meas_fp_mid[df_meas_fp_mid['partid'] == elt]['E_max'].iloc[0], df_meas_fp_post[df_meas_fp_post['partid'] == elt]['E_max'].iloc[0]]
    plot_mod_D_vec = [elt, df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['R_sys'].iloc[0], df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['R_sys'].iloc[0], df_mod_fp_post[df_mod_fp_post['partid'] == elt]['R_sys'].iloc[0], df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['C_ao'].iloc[0], df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['C_ao'].iloc[0], df_mod_fp_post[df_mod_fp_post['partid'] == elt]['C_ao'].iloc[0],  df_mod_fp_pre[df_mod_fp_pre['partid'] == elt]['E_max'].iloc[0], df_mod_fp_mid[df_mod_fp_mid['partid'] == elt]['E_max'].iloc[0], df_mod_fp_post[df_mod_fp_post['partid'] == elt]['E_max'].iloc[0]]
    
    df_meas_plot_D = df_meas_plot_D.append(dict(zip(plot_col, plot_meas_D_vec)), ignore_index=True)
    df_mod_plot_D = df_mod_plot_D.append(dict(zip(plot_col, plot_mod_D_vec)), ignore_index=True)
    


### COMPUTE CHANGE STATISTICS

# Filter out the participants with post CPET finger pressure measurements
print(CPET_pre_ids)
print(CPET_post_ids)

# Find the parcipants that have existing measurements 
# for CPET on both the first and final measurement days
df_meas_changes_fp = df_meas_changes_fp[[not np.isnan(ele) for ele in df_meas_changes_fp['R_sys_12wk_CPET']]]
df_mod_changes_fp = df_mod_changes_fp[[not np.isnan(ele) for ele in df_mod_changes_fp['R_sys_12wk_CPET']]]

ids_filtered = list(df_mod_changes_fp["partid"])

df_meas_changes_tono = df_meas_changes_tono[[(ele in ids_filtered) for ele in df_meas_changes_tono['partid']]]
df_mod_changes_tono = df_mod_changes_tono[[(ele in ids_filtered) for ele in df_mod_changes_tono['partid']]]


# Average change compilation - Model estimated parameters

# Post CPET results for model estimated parameters
# Mean of absolute changes and mean of signed changes 
df_meanvals_CPET = df_mod_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET",
                                      "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET",
                                      "R_sys_12wk_CPET", "C_ao_12wk_CPET", 
                                      "E_max_12wk_CPET"]].abs().mean()
df_meanvals_CPET_signed = df_mod_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET",
                                             "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET",
                                             "R_sys_12wk_CPET", "C_ao_12wk_CPET",
                                             "E_max_12wk_CPET"]].mean()

# Standard deviations
df_meanvals_CPET_std = df_mod_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET",
                                          "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET",
                                          "R_sys_12wk_CPET", "C_ao_12wk_CPET",
                                          "E_max_12wk_CPET"]].abs().std()
df_meanvals_CPET_std_signed = df_mod_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET",
                                                 "R_sys_post_CPET", "C_ao_post_CPET",
                                                 "E_max_post_CPET", 
                                                 "R_sys_12wk_CPET", "C_ao_12wk_CPET",
                                                 "E_max_12wk_CPET"]].std()


# Statistics for changes computed from the same participants but only with pre CPET measurements
df_meanvals_tono = df_mod_changes_tono[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                        "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                        "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].abs().mean()
df_meanvals_fp = df_mod_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                    "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                    "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].abs().mean()
df_meanvals_fp_std = df_mod_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                        "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                        "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].abs().std()
df_meanvals_fp_signed = df_mod_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                           "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                           "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].mean()
df_meanvals_fp_std_signed = df_mod_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                               "R_sys_12wk", "C_ao_12wk", "E_max_12wk",           
                                               "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].std()

print('MODEL')
print('-----------------------------')
print('---- R_sys -- C_ao -- E_max -')
print('-----------------------------')
print('CPET - pre                   ')
print(df_meanvals_CPET["R_sys_pre_CPET"], df_meanvals_CPET["C_ao_pre_CPET"],
      df_meanvals_CPET["E_max_pre_CPET"])
print(df_meanvals_CPET_std["R_sys_pre_CPET"], df_meanvals_CPET_std["C_ao_pre_CPET"],
      df_meanvals_CPET_std["E_max_pre_CPET"])
print('-----------------------------')
print('CPET - post                  ')
print(df_meanvals_CPET["R_sys_post_CPET"], df_meanvals_CPET["C_ao_post_CPET"],
      df_meanvals_CPET["E_max_post_CPET"])
print(df_meanvals_CPET_std["R_sys_post_CPET"], df_meanvals_CPET_std["C_ao_post_CPET"],
      df_meanvals_CPET_std["E_max_post_CPET"])
print('-----------------------------')
print('CPET - 12wk                  ')
print(df_meanvals_CPET["R_sys_12wk_CPET"], df_meanvals_CPET["C_ao_12wk_CPET"],
      df_meanvals_CPET["E_max_12wk_CPET"])
print(df_meanvals_CPET_std["R_sys_12wk_CPET"], df_meanvals_CPET_std["C_ao_12wk_CPET"],
      df_meanvals_CPET_std["E_max_12wk_CPET"])
print('-----------------------------')
print('FP - 12wk                    ')
print(df_meanvals_fp["R_sys_12wk"], df_meanvals_fp["C_ao_12wk"], df_meanvals_fp["E_max_12wk"])
print(df_meanvals_fp_std["R_sys_12wk"], df_meanvals_fp_std["C_ao_12wk"],
      df_meanvals_fp_std["E_max_12wk"])
print('-----------------------------')
print('TONO - 12wk                  ')
print(df_meanvals_tono["R_sys_12wk"], df_meanvals_tono["C_ao_12wk"], df_meanvals_tono["E_max_12wk"])
print('-----------------------------')
print('FP - 6wk                     ')
print(df_meanvals_fp["R_sys_6wk"], df_meanvals_fp["C_ao_6wk"], df_meanvals_fp["E_max_6wk"])
print('-----------------------------')
print('TONO - 6wk                   ')
print(df_meanvals_tono["R_sys_6wk"], df_meanvals_tono["C_ao_6wk"], df_meanvals_tono["E_max_6wk"])
print('-----------------------------')
print('MODEL SIGNED')
print('-----------------------------')
print('---- R_sys -- C_ao -- E_max -')
print('-----------------------------')
print('CPET - pre                   ')
print(df_meanvals_CPET_signed["R_sys_pre_CPET"], df_meanvals_CPET_signed["C_ao_pre_CPET"],
      df_meanvals_CPET_signed["E_max_pre_CPET"])
print(df_meanvals_CPET_std_signed["R_sys_pre_CPET"], df_meanvals_CPET_std_signed["C_ao_pre_CPET"],
      df_meanvals_CPET_std_signed["E_max_pre_CPET"])
print('-----------------------------')
print('CPET - post                  ')
print(df_meanvals_CPET_signed["R_sys_post_CPET"], df_meanvals_CPET_signed["C_ao_post_CPET"],
      df_meanvals_CPET_signed["E_max_post_CPET"])
print(df_meanvals_CPET_std_signed["R_sys_post_CPET"], df_meanvals_CPET_std_signed["C_ao_post_CPET"],
      df_meanvals_CPET_std_signed["E_max_post_CPET"])
print('-----------------------------')
print('CPET - 12wk                  ')
print(df_meanvals_CPET_signed["R_sys_12wk_CPET"], df_meanvals_CPET_signed["C_ao_12wk_CPET"],
      df_meanvals_CPET_signed["E_max_12wk_CPET"])
print('-----------------------------')
print('FP - 12wk                    ')
print(df_meanvals_fp_signed["R_sys_12wk"], df_meanvals_fp_signed["C_ao_12wk"],
      df_meanvals_fp_signed["E_max_12wk"])
print(df_meanvals_fp_std_signed["R_sys_12wk"], df_meanvals_fp_std_signed["C_ao_12wk"],
      df_meanvals_fp_std_signed["E_max_12wk"])
print('-----------------------------')
print('FP - 6wk                     ')
print(df_meanvals_fp_signed["R_sys_6wk"], df_meanvals_fp_signed["C_ao_6wk"], 
      df_meanvals_fp_signed["E_max_6wk"])
print('-----------------------------')

df_meanvals_CPET_signed.to_csv('CPETchanges/Means_mod_CPET_signed_CL.csv')
df_meanvals_CPET.to_csv('CPETchanges/Means_mod_CPET_CL.csv')



# Average change compilation - Conventionally estimated parameters

# Post CPET results for conventionally estimated parameters
# Mean of absolute changes and mean of signed changes
df_meanvals_CPET = df_meas_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET",
                                       "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET",
                                       "R_sys_12wk_CPET", "C_ao_12wk_CPET",
                                       "E_max_12wk_CPET"]].abs().mean()
df_meanvals_CPET_signed = df_meas_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET",
                                              "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET",
                                              "R_sys_12wk_CPET", "C_ao_12wk_CPET",
                                              "E_max_12wk_CPET"]].mean()
df_meanvals_CPET_std = df_meas_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET", "E_max_pre_CPET",
                                           "R_sys_post_CPET", "C_ao_post_CPET", "E_max_post_CPET",
                                           "R_sys_12wk_CPET", "C_ao_12wk_CPET",
                                           "E_max_12wk_CPET"]].abs().std()
df_meanvals_CPET_std_signed = df_meas_changes_fp[["R_sys_pre_CPET", "C_ao_pre_CPET",
                                                  "E_max_pre_CPET", 
                                                  "R_sys_post_CPET", "C_ao_post_CPET",
                                                  "E_max_post_CPET", 
                                                  "R_sys_12wk_CPET", "C_ao_12wk_CPET",
                                                  "E_max_12wk_CPET"]].std()



# Statistics for changes computed from the same participants but only with pre CPET measurements
df_meanvals_tono = df_meas_changes_tono[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                         "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                         "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].abs().mean()
df_meanvals_fp = df_meas_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                     "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                     "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].abs().mean()
df_meanvals_fp_std = df_meas_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                         "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                         "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].abs().std()
df_meanvals_fp_signed = df_meas_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                            "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 
                                            "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].mean()
df_meanvals_fp_std_signed = df_meas_changes_fp[["R_sys_6wk", "C_ao_6wk", "E_max_6wk", 
                                                "R_sys_12wk", "C_ao_12wk", "E_max_12wk",
                                                "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]].std()


print('MEAS')
print('-----------------------------')
print('---- R_sys -- C_ao -- E_max -')
print('-----------------------------')
print(df_meanvals_CPET["R_sys_pre_CPET"], df_meanvals_CPET["C_ao_pre_CPET"],
      df_meanvals_CPET["E_max_pre_CPET"])
print(df_meanvals_CPET_std["R_sys_pre_CPET"], df_meanvals_CPET_std["C_ao_pre_CPET"],
      df_meanvals_CPET_std["E_max_pre_CPET"])
print('-----------------------------')
print('CPET - post                  ')
print(df_meanvals_CPET["R_sys_post_CPET"], df_meanvals_CPET["C_ao_post_CPET"],
      df_meanvals_CPET["E_max_post_CPET"])
print(df_meanvals_CPET_std["R_sys_post_CPET"], df_meanvals_CPET_std["C_ao_post_CPET"],
      df_meanvals_CPET_std["E_max_post_CPET"])
print('-----------------------------')
print('CPET - 12wk                  ')
print(df_meanvals_CPET["R_sys_12wk_CPET"], df_meanvals_CPET["C_ao_12wk_CPET"],
      df_meanvals_CPET["E_max_12wk_CPET"])
print(df_meanvals_CPET_std["R_sys_12wk_CPET"], df_meanvals_CPET_std["C_ao_12wk_CPET"],
      df_meanvals_CPET_std["E_max_12wk_CPET"])
print('-----------------------------')
print('FP - 12wk                    ')
print(df_meanvals_fp["R_sys_12wk"], df_meanvals_fp["C_ao_12wk"], df_meanvals_fp["E_max_12wk"])
print(df_meanvals_fp_std["R_sys_12wk"], df_meanvals_fp_std["C_ao_12wk"],
      df_meanvals_fp_std["E_max_12wk"])
print('-----------------------------')
print('TONO - 12wk                  ')
print(df_meanvals_tono["R_sys_12wk"], df_meanvals_tono["C_ao_12wk"], df_meanvals_tono["E_max_12wk"])
print('-----------------------------')
print('FP - 6wk                     ')
print(df_meanvals_fp["R_sys_6wk"], df_meanvals_fp["C_ao_6wk"], df_meanvals_fp["E_max_6wk"])
print('-----------------------------')
print('TONO - 6wk                   ')
print(df_meanvals_tono["R_sys_6wk"], df_meanvals_tono["C_ao_6wk"], df_meanvals_tono["E_max_6wk"])
print('-----------------------------')
print('MEAS SIGNED')
print('-----------------------------')
print('---- R_sys -- C_ao -- E_max -')
print('-----------------------------')
print('CPET - pre                   ')
print(df_meanvals_CPET_signed["R_sys_pre_CPET"], df_meanvals_CPET_signed["C_ao_pre_CPET"],
      df_meanvals_CPET_signed["E_max_pre_CPET"])
print(df_meanvals_CPET_std_signed["R_sys_pre_CPET"], df_meanvals_CPET_std_signed["C_ao_pre_CPET"],
      df_meanvals_CPET_std_signed["E_max_pre_CPET"])
print('-----------------------------')
print('CPET - post                  ')
print(df_meanvals_CPET_signed["R_sys_post_CPET"], df_meanvals_CPET_signed["C_ao_post_CPET"],
      df_meanvals_CPET_signed["E_max_post_CPET"])
print(df_meanvals_CPET_std_signed["R_sys_post_CPET"], df_meanvals_CPET_std_signed["C_ao_post_CPET"],
      df_meanvals_CPET_std_signed["E_max_post_CPET"])
print('-----------------------------')
print('CPET - 12wk                  ')
print(df_meanvals_CPET_signed["R_sys_12wk_CPET"], df_meanvals_CPET_signed["C_ao_12wk_CPET"],
      df_meanvals_CPET_signed["E_max_12wk_CPET"])
print('-----------------------------')
print('FP - 12wk                    ')
print(df_meanvals_fp_signed["R_sys_12wk"], df_meanvals_fp_signed["C_ao_12wk"],
      df_meanvals_fp_signed["E_max_12wk"])
print(df_meanvals_fp_std_signed["R_sys_12wk"], df_meanvals_fp_std_signed["C_ao_12wk"],
      df_meanvals_fp_std_signed["E_max_12wk"])
print('-----------------------------')
print('FP - 6wk                     ')
print(df_meanvals_fp_signed["R_sys_6wk"], df_meanvals_fp_signed["C_ao_6wk"],
      df_meanvals_fp_signed["E_max_6wk"])
print('-----------------------------')

df_meanvals_CPET_signed.to_csv('CPETchanges/Means_meas_CPET_signed_CL.csv')
df_meanvals_CPET.to_csv('CPETchanges/Means_meas_CPET_CL.csv')

