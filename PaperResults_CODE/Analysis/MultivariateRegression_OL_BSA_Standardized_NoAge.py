# Figure regression variables - Open-loop cardiovascular model parameter estimation results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pingouin as pgn
import statsmodels.formula.api as smf
import statsmodels.api as sm


# Load Carotid Pressure Data

df_meas_tono = pd.read_pickle('Outputs_measurements_tono_BSA_WF.pkl')
df_mod_tono = pd.read_pickle('Outputs_models_tono_Open_BSA_WF.pkl')

# Load VO2max data

filepath = "../Data/PaperData_ReID.xlsx"
VO2table = pd.read_excel(filepath, engine='openpyxl')

# Check for invalid E_max in finger pressure measurement derived data
remove_idx = []
for idx, row in df_meas_tono.iterrows():
    if (row["E_max"] > 20.) or (row["E_max"] < 0.):
        print("Invalid E_max at (ID,idx):",row["id"],idx)
        remove_idx.append(idx)

print("Search for invalid E_max values complete")


# Split dataset by measurement day

df_meas_tono_pre = df_meas_tono[df_meas_tono['test_day'] == 'Pre-test day 2']
df_meas_tono_mid = df_meas_tono[df_meas_tono['test_day'] == 'Mid-test']
df_meas_tono_post = df_meas_tono[df_meas_tono['test_day'] == 'Post-test day 2']

df_mod_tono_pre = df_mod_tono[df_mod_tono['test_day'] == 'Pre-test day 2']
df_mod_tono_mid = df_mod_tono[df_mod_tono['test_day'] == 'Mid-test']
df_mod_tono_post = df_mod_tono[df_mod_tono['test_day'] == 'Post-test day 2']

VO2_pre = VO2table[VO2table['test_day'] == 'Pre-test day 2']
VO2_post = VO2table[VO2table['test_day'] == 'Post-test day 2'] 

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


# Set up data structures for storing the computed changes
change_col = ['partid', 'P_sys_6wk', 'P_dia_6wk', 'SV_6wk', "R_sys_6wk", "C_ao_6wk", "E_max_6wk", 'P_sys_12wk', 'P_dia_12wk', 'SV_12wk', "R_sys_12wk", "C_ao_12wk", "E_max_12wk", 'P_sys_612wk', 'P_dia_612wk', 'SV_612wk', "R_sys_612wk", "C_ao_612wk", "E_max_612wk"]

plot_col=["id", "R1", "R2", "R3", 
                "C1", "C2", "C3", 
                "E1", "E2", "E3", 
                "Rpers", "Cpers", "Epers", 
                'P0', 'P1', 'P2', 
                'Pd0', 'Pd1', 'Pd2',
                "SV0", "SV1", "SV2", 
                "VO2_A", "VO2_B", "delta_VO2", 
                "Age", "Sex", "height", 
                "wt0", "wt1", "wt2", 
                "BMI0", "BMI1", "BMI2",
                "T1", "T2", "T3"]
plot_col_mod=["id", "R1", "R2", "R3", 
                    "C1", "C2", "C3", 
                    "E1", "E2", "E3",
                    "Z1", "Z2", "Z3",
                    'tp1','tp2','tp3', 
                    "Rpers", "Cpers", "Epers", 
                    'P0', 'P1', 'P2', 
                    'Pd0', 'Pd1', 'Pd2',
                    "SV0", "SV1", "SV2", 
                    "VO2_A", "VO2_B", "delta_VO2", 
                    "Age", "Sex", "height",
                    "wt0", "wt1", "wt2", 
                    "BMI0", "BMI1", "BMI2",
                    "T1", "T2", "T3"]


df_meas_plot_C = pd.DataFrame(columns=["id", "R1", "R2", "R3", 
                                             "C1", "C2", "C3", 
                                             "E1", "E2", "E3", 
                                             "Rpers", "Cpers", "Epers", 
                                             'P0', 'P1', 'P2', 
                                             'Pd0', 'Pd1', 'Pd2',
                                             "SV0", "SV1", "SV2", 
                                             "VO2_A", "VO2_B", "delta_VO2", 
                                             "Age", "Sex", "height", 
                                             "wt0", "wt1", "wt2", 
                                             "BMI0", "BMI1", "BMI2",
                                             "T1", "T2", "T3"])
df_mod_plot_C = pd.DataFrame(columns=["id", "R1", "R2", "R3", 
                                            "C1", "C2", "C3", 
                                            "E1", "E2", "E3",
                                            "Z1", "Z2", "Z3",
                                            'tp1','tp2','tp3',
                                            "Rpers", "Cpers", "Epers", 
                                            'P0', 'P1', 'P2', 
                                            'Pd0', 'Pd1', 'Pd2',
                                            "SV0", "SV1", "SV2", 
                                            "VO2_A", "VO2_B", "delta_VO2", 
                                            "Age", "Sex", "height", 
                                            "wt0", "wt1", "wt2", 
                                            "BMI0", "BMI1", "BMI2",
                                            "T1", "T2", "T3"])




df_meas_changes_tono = pd.DataFrame(columns=change_col)
df_mod_changes_tono = pd.DataFrame(columns=change_col)


### LOOP THROUGH CAROTID MEASUREMENTS TO COMPUTE CHANGES AND COLLECTED DATA
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
    
    line_vec_meas = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                          R_sys_6wk, C_ao_6wk, E_max_6wk, 
                          P_sys_12wk, P_dia_12wk, SV_12wk, 
                          R_sys_12wk, C_ao_12wk, E_max_12wk, 
                          P_sys_612wk, P_dia_612wk, SV_612wk, 
                          R_sys_612wk, C_ao_612wk, E_max_612wk]
    
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
    
    line_vec_mod = [elt, P_sys_6wk, P_dia_6wk, SV_6wk, 
                          R_sys_6wk, C_ao_6wk, E_max_6wk, 
                          P_sys_12wk, P_dia_12wk, SV_12wk, 
                          R_sys_12wk, C_ao_12wk, E_max_12wk, 
                          P_sys_612wk, P_dia_612wk, SV_612wk, 
                          R_sys_612wk, C_ao_612wk, E_max_612wk]
    
    try:
        VO2_B = VO2_post[VO2_post['Partid'] == elt]['VO2max'].iloc[0]
        VO2_A = VO2_pre[VO2_pre['Partid'] == elt]['VO2max'].iloc[0]
    except:
        print(elt, "missing val VO2") 
        VO2_A = np.nan
        VO2_B = np.nan  
    
    try:
        new_elt = elt
        age = float(VO2Table[VO2Table['Partid'] == new_elt]['Age'].iloc[0])
        
    except:
        age = np.nan

    try:
        new_elt = elt
        sex = VO2table[VO2table['Partid'] == new_elt]['Sex'].iloc[0]
    except Exception as exc:
        print(exc) 
        sex = 'Unknown'
    
    height_tab = VO2table[VO2table['Partid'] == elt]
    height = height_tab[height_tab['Partid'] == 1]['height'].iloc[0]    
    
    
    wt0 = height_tab[height_tab['test_day'] == 1]['weight'].iloc[0]
    wt1 = height_tab[height_tab['test_day'] == 2]['weight'].iloc[0]
    wt2 = height_tab[height_tab['test_day'] == 3]['weight'].iloc[0]
    
    # Append
    df_meas_changes_tono = df_meas_changes_tono.append(dict(zip(change_col,line_vec_meas)),ignore_index=True)
    df_mod_changes_tono = df_mod_changes_tono.append(dict(zip(change_col,line_vec_mod)),ignore_index=True)

    plot_meas_C_vec = [elt, df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['R_sys'].iloc[0], 
                            df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['R_sys'].iloc[0], 
                            df_meas_tono_post[df_meas_tono_post['partid'] == elt]['R_sys'].iloc[0], 
                            df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['C_ao'].iloc[0], 
                            df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['C_ao'].iloc[0], 
                            df_meas_tono_post[df_meas_tono_post['partid'] == elt]['C_ao'].iloc[0],
                            df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['E_max'].iloc[0], 
                            df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['E_max'].iloc[0], 
                            df_meas_tono_post[df_meas_tono_post['partid'] == elt]['E_max'].iloc[0], 
                            (df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['R_sys'].iloc[0]+df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['R_sys'].iloc[0])/2., 
                            (df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['C_ao'].iloc[0]+df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['C_ao'].iloc[0])/2., 
                            (df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['E_max'].iloc[0]+df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['E_max'].iloc[0])/2., 
                            df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['P_sys'].iloc[0], 
                            df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['P_sys'].iloc[0], 
                            df_meas_tono_post[df_meas_tono_post['partid'] == elt]['P_sys'].iloc[0],
                            df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['P_dia'].iloc[0], 
                            df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['P_dia'].iloc[0], 
                            df_meas_tono_post[df_meas_tono_post['partid'] == elt]['P_dia'].iloc[0],
                            df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['SV'].iloc[0],
                            df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['SV'].iloc[0],
                            df_meas_tono_post[df_meas_tono_post['partid'] == elt]['SV'].iloc[0],
                            VO2_A, VO2_B, VO2_B-VO2_A,
                            age,sex, height, 
                            wt0, wt1, wt2, 
                            wt0/((height/100.)**2), wt1/((height/100.)**2), wt2/((height/100.)**2),
                            df_meas_tono_pre[df_meas_tono_pre['partid'] == elt]['T'].iloc[0],
                            df_meas_tono_mid[df_meas_tono_mid['partid'] == elt]['T'].iloc[0],
                            df_meas_tono_post[df_meas_tono_post['partid'] == elt]['T'].iloc[0]]
    
    plot_mod_C_vec = [elt, df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['R_sys'].iloc[0], 
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['R_sys'].iloc[0], 
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['R_sys'].iloc[0], 
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['C_ao'].iloc[0], 
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['C_ao'].iloc[0], 
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['C_ao'].iloc[0],  
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['E_max'].iloc[0], 
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['E_max'].iloc[0], 
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['E_max'].iloc[0],
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['Z_ao'].iloc[0], 
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['Z_ao'].iloc[0], 
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['Z_ao'].iloc[0],
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['t_peak'].iloc[0], 
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['t_peak'].iloc[0], 
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['t_peak'].iloc[0],
                           (df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['R_sys'].iloc[0]+df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['R_sys'].iloc[0])/2., 
                           (df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['C_ao'].iloc[0]+df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['C_ao'].iloc[0])/2.,
                           (df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['E_max'].iloc[0]+df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['E_max'].iloc[0])/2., 
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['P_sys'].iloc[0], 
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['P_sys'].iloc[0], 
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['P_sys'].iloc[0],
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['P_dia'].iloc[0], 
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['P_dia'].iloc[0], 
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['P_dia'].iloc[0],
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['SV'].iloc[0],
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['SV'].iloc[0],
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['SV'].iloc[0],
                           VO2_A, VO2_B, VO2_B-VO2_A,
                           age,sex, height, 
                           wt0, wt1, wt2, 
                           wt0/((height/100.)**2), 
                           wt1/((height/100.)**2), 
                           wt2/((height/100.)**2),
                           df_mod_tono_pre[df_mod_tono_pre['partid'] == elt]['T'].iloc[0],
                           df_mod_tono_mid[df_mod_tono_mid['partid'] == elt]['T'].iloc[0],
                           df_mod_tono_post[df_mod_tono_post['partid'] == elt]['T'].iloc[0]]
    
    




    df_meas_plot_C = df_meas_plot_C.append(dict(zip(plot_col, plot_meas_C_vec)), ignore_index=True)
    df_mod_plot_C = df_mod_plot_C.append(dict(zip(plot_col_mod, plot_mod_C_vec)), ignore_index=True)
    

# Post-process data, compute means, and prepare for regression analysis
df_par = df_mod_plot_C

# Compute population means across all measurement days from model based parameter estimates
popmeanR_mod_C = np.mean(np.append(np.append(df_par["R1"].values, df_par["R2"].values), df_par["R3"].values))
popmeanC_mod_C = np.mean(np.append(np.append(df_par["C1"].values, df_par["C2"].values), df_par["C3"].values))
popmeanE_mod_C = np.mean(np.append(np.append(df_par["E1"].values, df_par["E2"].values), df_par["E3"].values))

# Compute population means across all measurement days from conventional parameter estimates
df_par = df_meas_plot_C

popmeanR_meas_C = np.mean(np.append(np.append(df_par["R1"].values, df_par["R2"].values), df_par["R3"].values))
popmeanC_meas_C = np.mean(np.append(np.append(df_par["C1"].values, df_par["C2"].values), df_par["C3"].values))
popmeanE_meas_C = np.mean(np.append(np.append(df_par["E1"].values, df_par["E2"].values), df_par["E3"].values))


# Make a selection of relevant measurement data
df_reg_meas = df_meas_plot_C[["id", "SV0", "SV1", "SV2", "P0", "P1", "P2", "Pd0", "Pd1", "Pd2", "PAI_PP", "PAI_MP", "PAI_PM", "PAI_BA", "VO2_A", "VO2_B", "Sex", "Age", "delta_VO2", "wt0", "wt1", "wt2", "BMI0", "BMI1", "BMI2", "height", 'T1', 'T2', 'T3']]

# Make a selection of relevant cardiovascular model estimated data
df_reg_mod = df_mod_plot_C[["id", "R3", "R2", "R1", "C3", "C2", "C1", "E3", "E2", "E1","tp1","tp2","tp3", "Rpers", "Cpers", "Epers"]]

# Combine measurements into a common table
df_reg = df_reg_meas.join(df_reg_mod.set_index('id'), on='id')

# Compute additional changes to be computed
df_reg["Rpop"] = [popmeanR_mod_C]*len(df_reg.index)
df_reg["Epop"] = [popmeanE_mod_C]*len(df_reg.index)
df_reg["Cpop"] = [popmeanC_mod_C]*len(df_reg.index)
df_reg['delta_Psys'] = df_reg['P2'] - df_reg['P0']
df_reg['delta_Pdia'] = df_reg['Pd2'] - df_reg['Pd0']
df_reg['delta_SV'] = df_reg['SV2'] - df_reg['SV0']
df_reg['delta_R'] = df_reg['R3'] - df_reg['R1']
df_reg['delta_C'] = df_reg['C3'] - df_reg['C1']
df_reg['delta_E'] = df_reg['E3'] - df_reg['E1']
df_reg['delta_tp'] = df_reg['tp3'] - df_reg['tp1']
df_reg["delta_BMI"] = df_reg["BMI2"] - df_reg["BMI0"]

# Extras for regression to the mean computations
df_reg["R1minRpop"] = df_reg["R1"] - df_reg["Rpop"]
df_reg["C1minCpop"] = df_reg["C1"] - df_reg["Cpop"]
df_reg["E1minEpop"] = df_reg["E1"] - df_reg["Epop"]
df_reg["RpersRpop"] = df_reg["Rpers"] - df_reg["Rpop"]
df_reg["CpersCpop"] = df_reg["Cpers"] - df_reg["Cpop"]
df_reg["EpersEpop"] = df_reg["Epers"] - df_reg["Epop"]




# Standardize variables
df_reg_std = df_reg.select_dtypes(include=[np.number]).apply(stats.zscore, nan_policy='omit')
# Re-insert variables that are not to be standardized
df_reg_std["Sex"] = df_reg["Sex"]
df_reg_std["id"] = df_reg["id"]
df_reg_std["E1"] = df_reg["E1"]
df_reg_std["R1"] = df_reg["R1"]
df_reg_std["C1"] = df_reg["C1"]
df_reg_std["E2"] = df_reg["E2"]
df_reg_std["R2"] = df_reg["R2"]
df_reg_std["C2"] = df_reg["C2"]
df_reg_std["E3"] = df_reg["E3"]
df_reg_std["R3"] = df_reg["R3"]
df_reg_std["C3"] = df_reg["C3"]
df_reg_std["delta_R"] = df_reg["delta_R"]
df_reg_std["delta_C"] = df_reg["delta_C"]
df_reg_std["delta_E"] = df_reg["delta_E"]





# Determine the participants within the top quartile of VO2max changes

DVO2_75pecentile = df_reg["delta_VO2"].quantile(q=0.75)

print(df_reg["delta_VO2"])

print(DVO2_75pecentile)

df_VO2_filt = df_reg[df_reg["delta_VO2"] >= DVO2_75pecentile]

print("Top quartile of VO2 changes")
print(df_VO2_filt[["id","Sex"]])


# Reformat table to longformat
df_reg_longform3 = pd.DataFrame()
df_reg_longform3["id"] = np.append( np.append( df_reg['id'], df_reg['id'] ), df_reg['id'])
df_reg_longform3["Cao"] = np.append(np.append(df_reg['C1'], df_reg['C2']), df_reg['C3'])
df_reg_longform3["Rsys"] = np.append(np.append(df_reg['R1'], df_reg['R2']), df_reg['R3'])
df_reg_longform3["Emax"] = np.append(np.append(df_reg['E1'], df_reg['E2']), df_reg['E3'])
df_reg_longform3["Age"] = np.append(np.append(df_reg['Age'], df_reg['Age']), df_reg['Age'])
df_reg_longform3["BMI"] = np.append(np.append(df_reg['BMI0'], df_reg['BMI1']), df_reg['BMI2'])
df_reg_longform3["wt"] = np.append(np.append(df_reg['wt0'], df_reg['wt1']), df_reg['wt2'])
df_reg_longform3["Sex"] = np.append(np.append(df_reg['Sex'], df_reg['Sex']), df_reg['Sex'])
df_reg_longform3["height"] = np.append(np.append(df_reg['height'], df_reg['height']), df_reg['height'])
df_reg_longform3["VO2"] = np.append(np.append(df_reg['VO2_A'], np.array([np.nan]*len(df_reg['VO2_A']))), df_reg['VO2_B'])
df_reg_longform3["delta_VO2"] = np.append(np.append(df_reg['delta_VO2'], df_reg['delta_VO2']), df_reg['delta_VO2'])
df_reg_longform3["delta_BMI"] = np.append(np.append(df_reg['delta_BMI'], df_reg['delta_BMI']), df_reg['delta_BMI'])
df_reg_longform3["delta_Psys"] = np.append(np.append(df_reg['delta_Psys'], df_reg['delta_Psys']), df_reg['delta_Psys'])
df_reg_longform3["Psys"] = np.append(np.append(df_reg['P0'], df_reg['P1']), df_reg['P2'])
df_reg_longform3["Pdia"] = np.append(np.append(df_reg['Pd0'], df_reg['Pd1']), df_reg['Pd2'])
df_reg_longform3["PP"] = df_reg_longform3["Psys"] - df_reg_longform3["Pdia"]
df_reg_longform3["SV"] = np.append(np.append(df_reg['SV0'], df_reg['SV1']), df_reg['SV2']) 
df_reg_longform3["Day"] = np.append( np.append( ["Day1"]*len(df_reg["id"]), ["Day2"]*len(df_reg["id"]) ), ["Day3"]*len(df_reg["id"]) )
























# Function for maiking a pairplot of the regression data types
def print_lined_scatterplots(df, key='id', hue_in='Sex', special_ids=[], special_id_vars=[]):
    # Initialize useful variables
    N = len(df.columns)
    col_list = list(df.columns)
    str_counter=0
    str_marker = np.zeros(N)
    labdict={'Cao':'$C_{\mathrm{ao}}$',
          'Rsys':'$R_{\mathrm{sys}}$',
          'Emax':'$E_{\mathrm{max}}$',
          'Age':'Age',
          'BMI':'BMI',
          'SV':'SV',
          'VO2':'$\mathrm{VO2}_{\mathrm{max}}$'}

    
    #Remove columns which should not be plotted from list of data
    for idx, col in enumerate(df.columns):
        if isinstance(df[col].iloc[0], str): 
            str_counter += 1
            str_marker[0] = 1
            col_list.remove(col)
    
    # Initialize figure opbject
    N_dat = N-str_counter 
    fig, axs = plt.subplots(N_dat, N_dat, figsize=(22,22))#, sharex=True, sharey=True)
    
    # Find number of unique individuals
    uniqueIds = len(set(df[key].values))
    
    print(col_list)
    # Loop over grid using the columns of interesting data
    for x, colx in enumerate(col_list):
        for y, coly in enumerate(col_list):
            # On the diagonal plot KDE plots.
            if x == y:
                sb.kdeplot(df[colx], ax = axs[x][y], hue = df[hue_in], palette=['orange', 'blue'], legend=False)
                if y==0: 
                    axs[x][y].yaxis.label.set_size(14)
#                    axs[x][y]._shared_x_axes.join(axs[x][y], axs[N_dat-1][0])
#                else:
#                    axs[x][y]._shared_x_axes.join(axs[x][y], axs[N_dat-1][y])
                axs[x][y].set_xlabel(None)
                if x==N_dat-1: axs[x][y].set_xlabel(labdict[colx],fontsize=14)
            else:
                # Off diagonal, do scatter plots
                #axs[x][y].plot([0., 1.], [0., 1.])
                # Draw lines between the common participants
                for idx in range(0,uniqueIds):
                    temp_id = df[key].iloc[idx]
                    df_subset = df[df[key] == temp_id]
                    df_subset.sort_values(by="Day", ascending='True')
                    mask_x = np.isfinite(df_subset[colx])
                    mask_y = np.isfinite(df_subset[coly])
                    mask = mask_x & mask_y
                    if df_subset[hue_in].iloc[0] == 'Female':
                        axs[x][y].plot(df_subset[coly][mask], df_subset[colx][mask], '-', color = 'orange')
                    else:
                        axs[x][y].plot(df_subset[coly][mask], df_subset[colx][mask], '-', color = 'blue')
                    #if (coly == 'Cao') and (df_subset[coly][mask].iloc[0] >= 1.): print(df_subset[coly])
                    #if (coly == 'Cao'): print(df_subset[coly])
                # Put some finish on this
                # Apply markers for case analysis subjects given in special_ids
                for temp_id in list(special_ids.values):
                    df_subset = df[df[key] == temp_id]
                    df_subset.sort_values(by="Day", ascending='True')
                    mask_x = np.isfinite(df_subset[colx])
                    mask_y = np.isfinite(df_subset[coly])
                    mask = mask_x & mask_y
                    # Find non-finite data and nans and apply a mask to plot only the relevant data.
                    if (colx in special_id_vars) or (coly in special_id_vars):
                        if df_subset[hue_in].iloc[0] == 'Female':
                            axs[x][y].plot(df_subset[coly][mask], df_subset[colx][mask], '-',linewidth='3', color = 'orange')
                        else:
                            axs[x][y].plot(df_subset[coly][mask], df_subset[colx][mask], '-',linewidth='3', color = 'blue')
                        axs[x][y].plot(df_subset[coly][mask], df_subset[colx][mask], 'o', color = 'k')
                        df_subset_final = df_subset[df_subset["Day"] == 'Day3']
                        mask_x_final = np.isfinite(df_subset_final[colx])
                        mask_y_final = np.isfinite(df_subset_final[coly])
                        mask_final = mask_x_final & mask_y_final
                        axs[x][y].plot(df_subset_final[coly][mask_final], df_subset_final[colx][mask_final], 'D', linewidth='4', color = 'k')
                        
                # Set markers for remaining participants
                df_final = df[df["Day"] == 'Day3']
                for temp_id in df_final[key]:
                    if temp_id in special_ids.values: continue
                    
                    df_final_pers = df_final[df_final[key] == temp_id]
                    
                    mask_x_final = np.isfinite(df_final_pers[colx])
                    mask_y_final = np.isfinite(df_final_pers[coly])
                    mask_final = mask_x_final & mask_y_final
                    
                    # Find non-finite data and nans and apply a mask to plot only the relevant data.
                    if df_final_pers[hue_in].iloc[0] == 'Female':
                        axs[x][y].plot(df_final_pers[coly][mask_final], df_final_pers[colx][mask_final], 'D',linewidth='3', color = 'orange')
                    else:
                        axs[x][y].plot(df_final_pers[coly][mask_final], df_final_pers[colx][mask_final], 'D',linewidth='3', color = 'blue')
                
                
                #axs[x][y].legend([],[], frameon=False)
                if x == 0:
                    g = sb.scatterplot(x=df[coly], y=df[colx], ax=axs[x][y], hue=df[hue_in], palette=['orange', 'blue'], legend=False)
                    #g1 = sb.scatterplot(x=df_final[coly][mask_final], y=df_final[colx][mask_final], ax=axs[x][y], hue=df_final[hue_in], palette = ['orange','blue'], legend=False, markers='D', s=5)
                    axs[x][y]._shared_y_axes.join(axs[x][y], axs[0][1])
                    axs[x][y]._shared_x_axes.join(axs[x][y], axs[0][y])
                    axs[x][y].xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
                    axs[x][y].xaxis.offsetText.set_visible(False)
                    axs[x][y].set_xlabel(None)
                    axs[x][y].set_ylabel(None)
                    if y>1:
                        axs[x][y].yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
                        axs[x][y].yaxis.offsetText.set_visible(False)
                    if y == 1:
                        axs[x][y].set_ylabel(labdict[colx], fontsize=14)
                    
                elif y == 0:
                    g = sb.scatterplot(x=df[coly], y=df[colx], ax=axs[x][y], hue=df[hue_in], palette=['orange', 'blue'], legend=False)
                    #g1 = sb.scatterplot(x=df_final[coly][mask_final], y=df_final[colx][mask_final], ax=axs[x][y], hue=df_final[hue_in], palette = ['orange','blue'], legend=False, markers='D', s=5)
                    axs[x][y]._shared_x_axes.join(axs[x][y], axs[N_dat-1][0])
                    if x != N_dat-1: axs[x][y]._shared_y_axes.join(axs[x][y], axs[x,N_dat-1])
                    if x < N_dat-1:
                        axs[x][y].xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
                        axs[x][y].xaxis.offsetText.set_visible(False)
                    axs[x][y].set_xlabel(None)
                    axs[x][y].set_ylabel(labdict[colx],fontsize=14)
                    if x == N_dat-1: axs[x][y].set_xlabel(labdict[coly],fontsize=14)
                    #g = sb.scatterplot(x=df[coly], y=df[colx], ax=axs[x][y], hue=df[hue_in], palette=['orange', 'blue'], legend=False)
                    #axs[x][y]._shared_y_axes.join(axs[x][y], axs[x][N_dat-1])
                    #axs[x][y]._shared_x_axes.join(axs[x][y], axs[1][0])
                elif x != 0:
                    g = sb.scatterplot(x=df[coly], y=df[colx], ax=axs[x][y], hue=df[hue_in], palette=['orange', 'blue'], legend=False)
                    #g1 = sb.scatterplot(x=df_final[coly][mask_final], y=df_final[colx][mask_final], ax=axs[x][y], hue=df_final[hue_in], palette = ['orange','blue'], legend=False, markers='D', s = 5)
                    axs[x][y]._shared_y_axes.join(axs[x][y], axs[x][0])
                    axs[x][y]._shared_x_axes.join(axs[x][y], axs[0][x])
                    axs[x][y].set_ylabel(None)
                    if y>0:
                        axs[x][y].yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
                        axs[x][y].yaxis.offsetText.set_visible(False)
                        
                        #axs[x][y].xaxis.offsetText.set_visible(False)
                    if x<N_dat-1:
                        axs[x][y].xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
                        axs[x][y].xaxis.offsetText.set_visible(False)
                        axs[x][y].set_xlabel(None)
                        #axs[x][y].yaxis.offsetText.set_visible(False)
                    if x==N_dat-1: axs[x][y].set_xlabel(labdict[coly],fontsize=14)#axs[x][y].xaxis.label.set_size(14)
                # Final catch
                
                
                
    #axs[1][N_dat-1].set_ylim([0.2,1])#.plot([0,1],[0,1])
    fig.tight_layout()
    fig.legend(['Male','Female'], loc=(0.89,0.084), fontsize=22, framealpha=0.1)#, bbox_to_anchor=(0,0))

    return fig




# Save figure! 
fig = print_lined_scatterplots(df_reg_longform3[["id", "Day","Rsys", "Cao",
                                                 "Emax","BMI","SV","VO2","Sex"]],
                               key='id', 
                               hue_in='Sex', 
                               special_ids=df_VO2_filt["id"], 
                               special_id_vars=["Rsys", "Cao", "Emax","BMI","SV","VO2"])

fig.savefig('Pairplot_RegVars_OL_BSA_NoAge.pdf')



# Create longform table without measurement day 2 data
df_reg_longform2 = pd.DataFrame()
df_reg_longform2["id"] = np.append(df_reg['id'], df_reg['id'])
df_reg_longform2["Cao"] = np.append(df_reg['C1'], df_reg['C3'])
df_reg_longform2["Rsys"] = np.append(df_reg['R1'],  df_reg['R3'])
df_reg_longform2["Emax"] = np.append(df_reg['E1'],  df_reg['E3'])
df_reg_longform2["Age"] = np.append(df_reg['Age'],  df_reg['Age'])
df_reg_longform2["BMI"] = np.append(df_reg['BMI0'],  df_reg['BMI2'])
df_reg_longform2["wt"] = np.append(df_reg['wt0'],  df_reg['wt2'])
df_reg_longform2["Sex"] = np.append(df_reg['Sex'], df_reg['Sex'])
df_reg_longform2["height"] = np.append(df_reg['height'], df_reg['height'])
df_reg_longform2["VO2"] = np.append(df_reg['VO2_A'], df_reg['VO2_B'])
df_reg_longform2["delta_VO2"] = np.append(df_reg['delta_VO2'], df_reg['delta_VO2'])
df_reg_longform2["delta_BMI"] = np.append(df_reg['delta_BMI'], df_reg['delta_BMI'])
df_reg_longform2["delta_Psys"] = np.append(df_reg['delta_Psys'], df_reg['delta_Psys'])
df_reg_longform2["Psys"] = np.append(df_reg['P0'], df_reg['P2'])
df_reg_longform2["Pdia"] = np.append(df_reg['Pd0'], df_reg['Pd2'])
df_reg_longform2["PP"] = df_reg_longform2["Psys"] - df_reg_longform2["Pdia"]
df_reg_longform2["SV"] = np.append(df_reg['SV0'], df_reg['SV2']) 
df_reg_longform2["Day"] = np.append( ["Day1"]*len(df_reg["id"]), ["Day2"]*len(df_reg["id"]) )

# Standardize regressors

df_reg_longform3_std = df_reg_longform3[["Age", "BMI", "SV", "VO2"]].select_dtypes(include=[np.number]).apply(stats.zscore,nan_policy='omit')
df_reg_longform3_std["Sex"] = df_reg_longform3["Sex"]
df_reg_longform3_std["id"] = df_reg_longform3["id"]
df_reg_longform3_std["Emax"] = df_reg_longform3["Emax"]
df_reg_longform3_std["Rsys"] = df_reg_longform3["Rsys"]
df_reg_longform3_std["Cao"] = df_reg_longform3["Cao"]

df_reg_longform3_std["Rpers"] = df_reg_longform3["Rpop"]
df_reg_longform3_std["Cpers"] = df_reg_longform3["Cpop"]
df_reg_longform3_std["Epers"] = df_reg_longform3["Epop"]
df_reg_longform3_std["Rpers"] = df_reg_longform3["Rpers"]
df_reg_longform3_std["Cpers"] = df_reg_longform3["Cpers"]
df_reg_longform3_std["Epers"] = df_reg_longform3["Epers"]

