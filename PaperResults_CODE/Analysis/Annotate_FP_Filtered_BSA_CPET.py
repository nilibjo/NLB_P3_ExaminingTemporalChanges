### ALL PARAMS
import pandas as pd
import numpy as np
import os

import scipy_vewk3_MVP as models

closed_loop_base_pars = dict()

def shift_minimum(p,q,t):
    # Identify minimum
    min_ind = np.argmin(p)
    t_base = t[0]
    t=t-t[0]
    if (t[min_ind] < 0.65*t[-1]):
        p_temp = p.copy()
        q_temp = q.copy()
        p_slope = p[:min_ind]
        p_temp[len(p)-min_ind:] = np.append(p_slope[1:], p[min_ind])
        p_temp[:len(p)-min_ind] = p[min_ind:]

        p = p_temp.copy()
        q_temp[len(q)-min_ind:] = q[:min_ind]
        q_temp[:len(q)-min_ind] = q[min_ind:]
        q = q_temp.copy()
        
        return p, q, t+t_base
    else:
        return p, q, t+t_base

################## READ FILES
filepath = "../PostCPET/Data/PaperData_CPET_Compact.xlsx"
data_table = pd.read_excel(filepath, engine='openpyxl')
    
kolonner = ["id", "Partid", "test_day","E_max","E_min","C_ao","R_sys","V_tot","Z_ao","C_sv","R_mv","t_peak","stddevsq"]
out_kolonner = ["id", "Partid", "test_day", "SV", "SV_int", "Q_lvao_max", "V_dia", "V_sys", "P_sys", "P_dia", "P_sys_REAL", "P_dia_REAL", "P_sv_sys", "P_sv_dia", "MVP"]

compiled_participants = []

try:
    Zfile = '../PostCPET/ClosedLoop_FingerPressure/ParameterVisualization/ZaoFits.csv'
    Zdata = pd.read_csv(Zfile, names=['Number', 'ID', 'partid', 'test_day', 'zval', 'eval', 'vval'], sep=',')
except Exception as exc:
    print(exc)
    print("Reference file ZaoFits file with fixed parameter values from the CPET/ClosedLoop_FingerPressure folder!")


meas_kol = ['id','partid','test_day','P_sys', 'P_dia', 'SV', 'R_sys', 'E_max', 'E_max_90perc', 'C_ao', 'data_points', 'T']
mod_kol = ['id','partid','test_day','P_sys', 'P_dia', 'SV', 'MVP', 'R_sys', 'E_max', 'C_ao', 'C_sv', 'V_tot', 'Z_ao', 't_peak', 'T']
Dataframe_measurements = pd.DataFrame(columns=meas_kol)
Dataframe_modelestimates = pd.DataFrame(columns=mod_kol)


################## LOOP THROUGH 
for idx, row in data_table.iterrows():
    
    trial_id = row['Partid']
    patient_id = row['id']
    test_day = row['test_day']
    
    print(trial_id)
    
    if np.isnan(int(trial_id[1:])): continue
    if (len(compiled_participants) > 0) and (str(patient_id) in compiled_participants): 
        print("Skipping trial_id: "+str(patient_id)+", because already compiled")
        continue
    
    compiled_participants.append(patient_id)
    
    for filename in os.listdir('../PostCPET/ClosedLoop_FingerPressure/ParameterVisualization/'):
        
        if "MultiFitsRaw_" in filename and '_'+str(patient_id) in filename: 
            
            print("Treating id: "+str(patient_id))
            
            pfix, pid, test_day_file, partid = filename.split('_') 
            
            ### READ BSA DATA
            id_match = data_table['Partid'] == trial_id
            if ('pre' in filename) or ('pr3' in filename):
                day_match = data_table['test_day'] == '1CPET'
            elif ('post' in filename) or ('po3' in filename):
                day_match = data_table['test_day'] == '3CPET'
            
            line_id = id_match & day_match
            wt_frame = data_table.loc[line_id]
            weight = float(wt_frame['weight'].iloc[0])
            height = float(row["height"])
            # Compute BSA
            bsa = np.sqrt(height*weight/3600.)
            
            if np.isnan(bsa): 
                # get weight from measurement day 1 in case of missing data for a subsequent measurement.
                part_tab = data_table[data_table['Partid'] == row['partid']]
                day_1 = part_tab[part_tab['test_day'] == 1]
                weight = float(day_1['weight'])
                bsa = np.sqrt(height*weight/3600)
                
            print('bsa: '+str(bsa))
            
            ### READ PRESSURE DATA
            t = wt_frame['time_finger'].iloc[0]
            #print(t)
            t = np.fromstring(t[1:-1], sep=' ')
            #print(t)
            p = wt_frame['pressure_finger'].iloc[0]
            #print(p)
            p = np.fromstring(p[1:-1], sep=', ')
            #print(p)
            q = wt_frame['flow_LVOT_fingersync'].iloc[0]
            #print(q)
            q = np.fromstring(q[1:-1], sep=', ')
            #print(q)
            
            ### READ EMIN DATA
            id_match = Zdata['ID'] == str(pid)
            
            z_frame = Zdata.loc[id_match]
            e_val = z_frame['eval'].iloc[0] 
            
            t_r = t.copy()
            t_r = t_r-t_r[0]
            p_r = p.copy()
            q_r = q.copy()
            
            print("Data loaded")
            T_r = np.round(t_r[-1],2)
            closed_loop_base_pars["T"] = T_r
            
            
            ### READ SV AND ESV DATA
            SV_raw = float(wt_frame["SV(4D)"].iloc[0])
            if SV_raw == '#VALUE!':
                SV_raw = float(wt_frame["SV_LVOT"].iloc[0])
            elif (SV_raw is None) or np.isnan(SV_raw):
                SV_raw = float(wt_frame["SV_LVOT"].iloc[0])
            
            
            ES_V_raw = float(wt_frame['LVESV(4D)'].iloc[0])
            if ES_V_raw == "#VALUE!":
                ES_V_raw = np.nan
            elif (ES_V_raw is None) or np.isnan(ES_V_raw):
                ES_V_raw = np.nan
            
            print(partid, pid, test_day_file)
            
            chosenfilename = filename
            skipflag = False
            Data = pd.read_csv('../PostCPET/ClosedLoop_FingerPressure/ParameterVisualization/'+filename,header=0,delimiter=',')            
                   
            key_filt = Data['stddevsq'] <= Data['stddevsq'].mean() 
            temp_frame = Data.loc[key_filt]
            Data_mean = temp_frame
            
            temp_list = Data['E_max']
            temp_len = len(list(temp_list))
            
            if test_day_file == 'pre': tday = 'Pre'
            if test_day_file == 'mid': tday = 'Mid'
            if test_day_file == 'post': tday = 'Post'
            if test_day_file == 'pr3': tday = '1CPET'
            if test_day_file == 'po3': tday = '3CPET'
            
            if (len(list(Data['E_max'])) != 0) and not (np.isnan(Data_mean.mean()).any()): 
                par_dict = Data_mean.mean().to_dict()
                par_dict['T'] = T_r
                par_dict['E_min'] = e_val
                par_dict['R_mv'] = 0.02
                
                vewk3 = models.VaryingElastance()
                vewk3.set_pars(**par_dict)
                ret_dict, t_eval,_ = models.solve_to_steady_state(vewk3, n_cycles=10)

                model_out_list = [pid, partid[:-4], tday,
                                  ret_dict["P_sys"], ret_dict["P_dia"], ret_dict["SV"]/bsa,  
                                  ret_dict['MVP'], par_dict['R_sys']/bsa, par_dict['E_max']/bsa, 
                                  par_dict['C_ao']/bsa,par_dict['C_sv']/bsa,
                                  par_dict['V_tot']/bsa, par_dict['Z_ao']/bsa, 
                                  par_dict['t_peak'], par_dict['T']]
                model_out_dict = dict(zip(mod_kol, model_out_list)) 

                rsys_est = np.mean(p_r)/((SV_raw/T_r)*bsa)
                cao_est = SV_raw/((np.max(p_r)-np.min(p_r))*bsa)
                emax_est = np.max(p_r)/(ES_V_raw*bsa)
                emax_90perc_est = 0.9*np.max(p_r)/(ES_V_raw*bsa)
                
                meas_out_list = [pid, partid[:-4], tday, 
                                 np.max(p_r), np.min(p_r), SV_raw/bsa, 
                                 rsys_est, emax_est, emax_90perc_est, cao_est, 
                                 len(p_r), T_r]
                meas_out_dict = dict(zip(meas_kol, meas_out_list))
                
                Dataframe_measurements = Dataframe_measurements.append(meas_out_dict, ignore_index=True)
                Dataframe_modelestimates = Dataframe_modelestimates.append(model_out_dict, ignore_index=True)
                
    
Dataframe_measurements.to_csv('Outputs_measurements_FP_BSA_CPET.csv')
Dataframe_measurements.to_pickle('Outputs_measurements_FP_BSA_CPET.pkl')

Dataframe_modelestimates.to_csv('Outputs_models_FP_BSA_CPET.csv')
Dataframe_modelestimates.to_pickle('Outputs_models_FP_BSA_CPET.pkl')

