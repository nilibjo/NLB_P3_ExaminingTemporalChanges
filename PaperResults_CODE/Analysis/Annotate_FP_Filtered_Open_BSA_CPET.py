import pandas as pd
import numpy as np
import os

import scipy_vewk3_openloop as models

closed_loop_base_pars = dict()

closed_loop_base_pars_short = dict()

def shift_minimum(p,q,t):
    #plt.plot(t,p,label='Old p')
    #plt.plot(t,q,label='New q')
    #plt.show()
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
filepath = "../postCPET/Data/PaperData_CPET_Compact.xlsx"
finometer_index = pd.read_excel(filepath, engine='openpyxl')

compiled_participants = []

try:
    Zfile = '../../ParamEst_V22_June/OpenLoopEstimations/ScaledFinger_FixedLimit_PsvFix_CPET/ParameterVisualization/ZaoFits.csv'
    Zdata = pd.read_csv(Zfile, names=['Number', 'ID', 'partid', 'test_day', 'zval', 'eval', 'vval'], sep=',')
except Exception as exc:
    print(exc)
    print("Reference file ZaoFits file with fixed parameter values from the postCPET/OpenLoop_CarotidPressure folder!")


mod_kol = ['id','partid','test_day',
           'P_sys', 'P_dia', 'SV', 
           'R_sys', 'E_max', 'E_min', 'C_ao', 'Z_ao' 't_peak', 
           'T']
Dataframe_modelestimates = pd.DataFrame(columns=mod_kol)

for idx, row in data_table.iterrows():
    
    trial_id = row['Partid']
    patient_id = row['ID']
    test_day = row['test_day']
    
    print(trial_id)
    
    if np.isnan(int(trial_id[1:])): continue
    if len(compiled_participants) > 0 and str(patient_id) in compiled_participants: 
        print("Skipping trial_id: "+str(patient_id)+", because already compiled")
        continue
    
    compiled_participants.append(patient_id)
    
    for filename in os.listdir('../OpenLoop_FingerPressure/ParameterVisualization/'):
        
        if "MultiFitsRaw_" in filename and '_'+str(patient_id) in filename: 
            
            print("Treating id: "+str(patient_id))
            
            pfix, pid, test_day_file, partid = filename.split('_') 
            
            ### READ BSA DATA
            id_match = data_table['Partid'] == trial_id
            if ('pre' in filename) or ('pr3' in filename):
                day_match = data_table['test_day'] == 1
            elif ('post' in filename) or ('po3' in filename):
                day_match = data_table['test_day'] == 3
            elif 'mid' in filename:
                day_match = data_table['test_day'] == 2
            
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
                bsa = np.sqrt(height*weight/3600.)
                
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
            
            
            print(partid, paistr+'_'+paiid, test_day_file)
            
            chosenfilename = filename
            skipflag = False
            Data = pd.read_csv('../OpenLoopEstimations/ScaledFinger_FixedLimit_PsvFix_CPET_Realigned_Nocutoff_Noskip/ParameterVisualization/'+filename,header=0,delimiter=',')            
                   
            key_filt = Data['stddevsq'] <= Data['stddevsq'].mean() 
            temp_frame = Data.loc[key_filt]
            Data_mean = temp_frame
            
            temp_list = Data['E_max']
            temp_len = len(list(temp_list))
            
            if test_day_file == 'pre': tday = 'Pre'
            if test_day_file == 'mid': tday = 'Mid'
            if test_day_file == 'post': tday = 'Post'
            if test_day_file == 'pr3': tday = 'CPET1'
            if test_day_file == 'po3': tday = 'CPET3'
            
            if (len(list(Data['E_max'])) != 0) and not (np.isnan(Data_mean.mean()).any()): 
                par_dict = Data_mean.mean().to_dict()
                par_dict['T'] = T_r
                par_dict['E_min'] = e_val
                par_dict['R_mv'] = 0.02
                
                
                vewk3 = models.VaryingElastance()
                vewk3.set_pars(**par_dict)
                ret_dict, t_eval,_ = models.solve_to_steady_state(vewk3, n_cycles=10)
                
                model_out_list = [pid, partid[:-4], tday, ret_dict["P_sys"], ret_dict["P_dia"], ret_dict["SV"]/bsa, par_dict['R_sys']/bsa, par_dict['E_max']/bsa, par_dict['E_min']/bsa, par_dict['C_ao']/bsa, par_dict['Z_ao']/bsa, par_dict['t_peak'], par_dict['T']]
                
                model_out_dict = dict(zip(mod_kol, model_out_list)) 
                
                Dataframe_modelestimates = Dataframe_modelestimates.append(model_out_dict, ignore_index=True)
                


Dataframe_modelestimates.to_csv('Outputs_models_FP_Open_BSA_CPET.csv')
Dataframe_modelestimates.to_pickle('Outputs_models_FP_Open_BSA_CPET.pkl')

