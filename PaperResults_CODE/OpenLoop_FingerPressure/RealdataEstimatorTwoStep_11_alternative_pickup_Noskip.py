import numpy as np
import scipy.optimize as opt

import pandas as pd
import csv

import matplotlib.pyplot as plt
import scipy_vewk3_openloop as models

import time

# Placeholder parameters, mainly taken from Segers et al Hypertension 2000
closed_loop_base_pars ={'C_ao': 1.13, 
                         'E_max': 1.5, 
                         'E_min': 0.03,#0.02
                         'R_mv': 0.02,
                         'R_sys': 1.11,
                         'T': 0.94,
                         'Z_ao': 0.033,
                         't_peak': 0.32
                        }


def shift_minimum(p,q,t):
    # Identify minimum
    min_ind = np.argmin(p)
    t_base = t[0]
    t=t-t[0]
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


def data_cost_function(pars, measurements=dict(P_sys=120, P_dia=80), active_pars=None, ret_all=False,
                base_pars=dict(closed_loop_base_pars)):
    """ general purpose wrapper for optimization with both lmfit and scipy"""
    measurement_scales = dict(P_sys=120, 
                              P_dia=80,
                              P_meas=100,
                              P_ao=100,
                              Q_lvao=500,
                              Q_aosv=500,
                              Q_svlv=300,
                              SV=100,
                              PP=40,
                              V_lv=100,
                              MVP=5.)
    
    it_base_pars = dict(base_pars)
    for idx, name in enumerate(active_pars):
        try: #LMFIT
            it_base_pars[name] = pars[name] 
        except: #SCIPY
            it_base_pars[name] = pars[idx]

    ve_closed = models.VaryingElastance()
    ve_closed.set_pars(**it_base_pars)
    var_dict, t_eval, scipy_sol = models.solve_to_steady_state(ve_closed,n_cycles=10,
                                                               n_eval_pts=len(measurements["P_ao"]))
    ret_dict = models.calc_summary(var_dict)
    residual = []
    residual_short = []
    for name, val in measurements.items():
        #if((name in ret_dict) and (not (name in var_dict))):
        if ((name in ret_dict) and (name == 'MVP')):
            residual_short.append((val - ret_dict[name])*(len(measurements["P_ao"])/40.)*2.5/(measurement_scales[name]))
        elif ((name in ret_dict) and (ret_dict[name].size == 1)):
            residual_short.append((val - ret_dict[name])*(len(measurements["P_ao"])/40.)*7.5/(measurement_scales[name]))
        else:
            pn, qn, tn = shift_minimum(var_dict["P_ao"], var_dict["Q_lvao"], t_eval)
            if name == 'P_ao':
                try:
                    residual.append((val - pn)/measurement_scales[name])
                except KeyError:
                    print("Skipping pressure measurement")
            elif name == 'Q_lvao':
                try:
                    residual.append((val - qn)/measurement_scales[name])
                except KeyError:
                    print("Skipping flow measurement")
            else:
                try:
                    residual.append((val - ret_dict[name])/measurement_scales[name])
                except KeyError:
                    residual.append((val - var_dict[name])/measurement_scales[name])
    
    residual = np.array(residual).flatten()
    residual = np.append(residual, residual_short)
    
    if ret_all:
        return residual, var_dict, t_eval, ret_dict
    else:
        return residual


def run_realdata_measurements(active_params,x0,writers,wflag,t_r,p_r,q_r,v_r,b_in,b_low):
    """ example trying to fit parameters with only 'clinically' measureable data """
    
    ############################################################# RESTING
    Psys_ref = np.max(p_r)
    Pdia_ref = np.min(p_r)
    SVref = np.trapz(q_r,t_r)
    measurements = dict(P_ao=p_r,Q_lvao=q_r, SV=SVref, P_sys=Psys_ref, P_dia=Pdia_ref)
    
    residual, var_dict, t_eval, ret_dict = data_cost_function(x0,
                                                         active_pars=active_params,
                                                         measurements=measurements, ret_all=True,
                                                         base_pars=closed_loop_base_pars)
    
    
    resultsR = opt.least_squares(data_cost_function, x0,
                                method='trf',
                                xtol=2.3e-16,ftol=2.3e-16,gtol=2.3e-16,diff_step=1e-3,
                                bounds=(b_low,b_in),
                                kwargs=dict(active_pars=active_params, measurements=measurements,base_pars=closed_loop_base_pars))
    
    residual, var_dict, t_eval, ret_dict = data_cost_function(resultsR.x, 
                                                         active_pars=active_params,
                                                         measurements=measurements, ret_all=True,
                                                         base_pars=closed_loop_base_pars)
    print("Estimated resting parameters")
    standard_dev_sq = np.sum(residual**2)/len(residual)
    print(active_params)
    estimated_pars = resultsR.x
    print("Normalized sum of squared residuals: ",standard_dev_sq)
    print(estimated_pars)
    
    estimated_pars = np.append(np.round(estimated_pars,4),standard_dev_sq)
    if wflag:
        writers[0].writerow(convert_to_sci(estimated_pars,4))

    return resultsR.x, standard_dev_sq, residual



def multi_sample(active, x0_0, writers,wflag,t_r,p_r,q_r,v_r,b_in,b_low,number,idnum,seqnum):
    
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    
    MultiFile = open('MultiFits-'+str(int(seqnum))+'-'+idnum+'-'+timestamp+'.csv','a')
    writerEst = csv.writer(MultiFile)
    writerEst.writerow(active)
    
    np.random.seed(112233)    
    minresidual = 10.**6
    x0min = []
    for i in range(0,number):
        print("Iteration #{} initiated".format(i))
        
        # Sample random inputvectors
        x0 = np.random.uniform(b_low,b_in)
               
        active_params = active.copy()
        # Fit the case
        estimateR = []
        try: 
            fig = plt.figure(i+1, figsize=(4,3), dpi=300)
            estimateR, stdevsq, residual = run_realdata_measurements(active_params, x0 ,writers,wflag,t_r,p_r,q_r,v_r,b_in,b_low)
            
            if stdevsq < minresidual:
                minresidual = stdevsq
                x0min = x0
            # Convert new parameter estimates in dicts
            it_base_pars = dict(closed_loop_base_pars)
            for idx, name in enumerate(active_params):
                it_base_pars[name] = estimateR[idx]
            ve_closed = models.VaryingElastance()
            ve_closed.set_pars(**it_base_pars)
            #Compute resting solution
            var_dict_r, t_eval_r, scipy_sol_r = models.solve_to_steady_state(ve_closed, 
                                                                   n_cycles=10,n_eval_pts=len(p_r))
            ret_dict_r = models.calc_summary(var_dict_r)
            print('Stroke Volume Rest:', ret_dict_r["SV"])
            
            plt.plot(t_eval_r-t_eval_r[0], var_dict_r["P_ao"], label="Rest #"+str(i))
            plt.plot(t_r, p_r, label="Rest data")
            plt.title('Fitting procedure #'+str(i))
            plt.xlabel('t [s]')
            plt.ylabel('p [mmHg]')
            plt.legend()
            fig.savefig('MultiEstimate'+str(i), pad_inches = 0.2)
            plt.close()
            print("Iteration #"+str(i)+": ")
            print(active)
            print(x0)
            print(estimateR)
            
            writerEst.writerow([i])
            writerEst.writerow(x0)
            writerEst.writerow(np.append(estimateR,stdevsq))
            
            print(ret_dict_r["SV"], ret_dict_r["P_map"], ret_dict_r["PP"], ret_dict_r["V_sys"], ret_dict_r["V_dia"], ret_dict_r["stroke_work_1"])
        except Exception as exc:
            print(exc)
            print("Timeout for iteration # {}".format(i))
            writerEst.writerow([i])
            writerEst.writerow(x0)
            plt.clf()
            plt.close()
            continue
        
    MultiFile.close()
    
    loop_lim = 20
    if len(x0min) == 0: 
        loop_lim = 1
        x0min = x0

    bl = np.array(x0min)*0.9
    bh = np.array(x0min)*1.1
    ##############
    #Round2
    ##############
    MultiFile = open('MultiFitsRound2-'+str(int(seqnum))+'-'+idnum+'-'+timestamp+'.csv','a')
    writerEst = csv.writer(MultiFile)
    writerEst.writerow(active)
    
    for i in range(0,loop_lim):
        print("Iteration #{} initiated".format(i))
        
        # Sample random input vectors
        
        x0 = np.random.uniform(bl, bh)
        x0 = np.min([x0,b_in],axis=0)
        x0 = np.max([x0,b_low],axis=0)
              
        active_params = active.copy()
        # Fit the case
        estimateR = []
        try: 
            fig = plt.figure(i+1, figsize=(4,3), dpi=300)
            estimateR, stdevsq, residual = run_realdata_measurements(active_params, x0,writers,wflag,t_r,p_r,q_r,v_r,b_in,b_low)
            
            # Convert new parameter estimates in dicts
            it_base_pars = dict(closed_loop_base_pars)
            for idx, name in enumerate(active_params):
                it_base_pars[name] = estimateR[idx]
            ve_closed = models.VaryingElastance()
            ve_closed.set_pars(**it_base_pars)
            #Compute resting solution
            var_dict_r, t_eval_r, scipy_sol_r = models.solve_to_steady_state(ve_closed, 
                                                                   n_cycles=10,n_eval_pts=len(p_r))
            ret_dict_r = models.calc_summary(var_dict_r)
            print('Stroke Volume Rest:', ret_dict_r["SV"])
            
            # Plot shifted pressure and estimated parameters
            
            plt.plot(t_eval_r-t_eval_r[0], var_dict_r["P_ao"], label="Rest #"+str(i))
            plt.plot(t_r, p_r, label="Rest data")
            plt.title('Fitting procedure #'+str(i))
            plt.xlabel('t [s]')
            plt.ylabel('p [mmHg]')
            plt.legend()
            fig.savefig('MultiEstimateStepTwo'+str(i), pad_inches = 0.2)
            plt.close()
            print("Iteration #"+str(i)+": ")
            print(active)
            print(x0)
            print(estimateR)
            
            writerEst.writerow([i])
            writerEst.writerow(x0)
            writerEst.writerow(np.append(estimateR,stdevsq))
            # Write 
            print(ret_dict_r["SV"], ret_dict_r["P_map"], ret_dict_r["PP"], ret_dict_r["V_sys"], ret_dict_r["V_dia"], ret_dict_r["stroke_work_1"])
        except Exception as exc:
            print(exc)
            print("Timeout for iteration # {}, round 2".format(i))
            writerEst.writerow([i])
            writerEst.writerow(x0)
            plt.clf()
            plt.close()
            continue
        
    MultiFile.close()
    
    try:
        del ve_closed
    except:
        print("Could not delete")
    return 0


def convert_to_sci(array, decimals):
    string = "{:."+str(decimals)+"e}"
    output = [string.format(e) for e in array]
    return output

def volume_initial_guess_estimator(flow, pressure, t):
    SV = np.trapz(flow, t)
    PP = np.max(pressure)-np.min(pressure)
    Cao = SV/PP # [mL/mmHg]
    ##
    MAP = np.mean(pressure)
    Rpz = MAP/(SV/(t[-1]-t[0]))#*60.0)*t[-1]
    return SV, Cao, Rpz

def is_input_none_nan(ref):
    boolval = not (isinstance(ref, str) or isinstance(ref, np.ndarray))
    return boolval
    
def is_array_none_nan(ref):
    boolval = not isinstance(ref, np.ndarray)
    return boolval

if __name__ == "__main__":
    ### READ DATA FILE
    
    data_path = "../Data/PaperData_Cleaned_ReID.xlsx"
    data_table = pd.read_excel(data_path, engine='openpyxl')
    completed = []   
    
    # Set up files for monitoring progression and extra information about fixed parameters
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    
    id_file = open('id_parameter_sequence-1-'+timestamp+'.csv','w')
    id_writer = csv.writer(id_file)
    
    ZFile = open('ZaoFits-1-'+timestamp+'.csv','a')
    writerZ = csv.writer(ZFile)
    id_writer.writerow(['Sequence_number', 'ID', 'Partid', 'test_day'])
    
    for idx, row in data_table.iterrows():
        
        if row['ID'] in completed:
            continue
        else:
            completed.append(row['ID'])
        
        
        # READ TIME SERIES DATA
        t = row["time_finger"]
        p = row["pressure_finger"]
        q = row["flow_LVOT_fingersync"]
        
        # Skip iteration if there is no waveform data avialable for the participant
        if is_input_none_nan(t) or is_input_none_nan(p) or is_input_none_nan(q):
            print("INVALID TIME SERIES INPUT FOUND", row['ID'])
            continue
        
        t = np.fromstring(t[1:-1], sep=' ')
        p = np.fromstring(p[1:-1], sep=', ')
        q = np.fromstring(q[1:-1], sep=', ')
        
        # Skip iteration if there is wrongly formatted waveform data for the participant
        if is_array_none_nan(t) or is_array_none_nan(p) or is_array_none_nan(q):
            print("INVALID TIME SERIES INPUT FOUND", row['ID'])
            continue
        
        print('Check2')
        
        t_r = df["time"]
        t_r = t_r-t_r[0]
        p_r = df["pressure"]
        q_r = df["flow"]
        v_r = None
        
        print("Data loaded")
        T_r = np.round(t_r[-1],2)
        closed_loop_base_pars["T"] = T_r
    
        # Resting resolution
        print("Resting resolution: ", t_r[1]-t_r[0])
        print("Number of datapoints in pressure and flow waveforms: ", len(p_r), len(q_r))
    
        print(len(p_r))
        p_temp = p_r.copy()
        q_temp = q_r.copy()
        t_temp = t_r.copy()

        ### READ BSA DATA
        weight = row["weight"]
        height = row["height"]
        # Compute BSA
        bsa = np.sqrt(height*weight/3600.)
        
        print("BSA [m²]: ", bsa)
        
        if np.isnan(bsa): 
            # get weight from measurement day 1 in case of missing data for a subsequent measurement.
            part_tab = data_table[data_table['Partid'] == row['partid']]
            day_1 = part_tab[part_tab['test_day'] == 1]
            weight = float(day_1['weight'])

            bsa = np.sqrt(height*weight/3600)
            print("New BSA: %.2f" % (bsa))
        
        active = ["C_ao", "R_sys", "Z_ao"]    
        x0 = np.array([0.436*bsa, 0.91725/bsa, 0.033])
        x_scale = [2.0, 2.0, 0.01]
        b_temp_in = [2.256, 2.963, 1.]
        b_temp_low = [0.148, 0.917/bsa, 0.001]
        
        if(np.max(p_r) > 140):
            Z_ao = 0.035
            E_min = 0.06
        else:
            Z_ao = 0.033
            E_min = 0.055
        _,C_ao,Rpz = volume_initial_guess_estimator(q_r, p_r, t_r) 
        R_sys = Rpz-Z_ao

        closed_loop_base_pars["C_ao"] = C_ao
        closed_loop_base_pars["Z_ao"] = Z_ao
        closed_loop_base_pars["R_sys"] = R_sys
        closed_loop_base_pars["E_min"] = E_min
        print(closed_loop_base_pars["T"])  
        
        print("FIVE PARAMETER FITS-------")
        print("---------RESTING----------")
        active =         ["E_max", "C_ao", "R_sys", "Z_ao", "t_peak"]    
        x0_0 = np.array( [7.58/bsa, C_ao, R_sys, Z_ao, 0.3975])
        
        b_in =           [10.84/bsa, 3.0, 2.963, 0.2, np.min([0.442,T_r])]
        b_low = np.array([0.5, 0.148, 0.917/bsa, 0.001, np.min([0.15,0.9*T_r])])
        
        # Run estimates
        multi_sample(active, x0_0, None,False,t_r,p_r,q_r,v_r,b_in,b_low,30,row['ID'],idx)
        # Write metadata and fixed parameters to file
        id_writer.writerow([idx, row['ID'], row['Partid'], row['test_day']])
        writerZ.writerow([idx, row['ID'],  row['Partid'], row['test_day'],Z_ao, E_min])

    id_file.close()
    ZFile.close()

