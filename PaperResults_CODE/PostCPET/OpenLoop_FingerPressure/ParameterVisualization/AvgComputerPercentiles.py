# Reformat ouput files
import pandas as pd
import numpy as np
import os
    
################## READ FILES
filepath = "../../Data/PaperData_CPET_ReID_Compact.xlsx"
data_table = pd.read_excel(filepath, engine='openpyxl')

compiled_participants = []

################## LOOP THROUGH 
for idx, row in finometer_index.iterrows():
    
    trial_id = row['Partid']
    patient_id = row['ID']
    test_day = row['test_day']
    
    if (patient_id in compiled_participants) or (not ('CPET' in str(test_day))):
        continue
    else:
        compiled_participants.append([patient_id])

    skipflag = False
    for filename in os.listdir(os.getcwd()):
        
        if "MultiFitsRound2" in filename and str(patient_id) in filename: 
            print("Treating id: "+str(patient_id))
            chosenfilename = filename
            skipflag = False
            break
        else:
            skipflag = True
    
    if skipflag: 
        print("Skip: "+str(patient_id))
        continue
    
    kolonner6 = ["E_max","C_ao","R_sys","Z_ao","t_peak","stddevsq"]
    
    kolonner = kolonner6
    pars = 5
    print(kolonner)
    
    # Read data from chosen file
    Data = pd.read_csv(chosenfilename,header=None,names=range(0,pars+1),skiprows=1,delimiter=',')
    
    # Set up an array for collection of relevant lines of output
    newframe = np.zeros([20, pars+1])
    print(newframe.size)
    
    # Set up arays for summation of parameter statistics
    sumrow1 = np.zeros(len(Data.loc[[1]].values.flatten()))
    sumrow2 = np.zeros(len(Data.loc[[1]].values.flatten()))
    varrow1 = np.zeros(len(Data.loc[[1]].values.flatten()))
    varrow2 = np.zeros(len(Data.loc[[1]].values.flatten()))
    
    # Pick only the lines of estimates, not from initial guesses
    sumcounter = 0
    count = -1
    nfcount = 1
    for index in Data.index[0:-2]:
        # Skip line if the current iteration only has a in initial estimate
        if index <= count: continue
        
        # Skip forward if the data point of the initial guess resulted in an incomplete attempt
        if np.isnan(Data.loc[[index]][1].values[0]) and np.isnan(Data.loc[[index+2]][1].values[0]):
            count = index + 1
        # Collect data if there exists an estimate
        elif np.isnan(Data.loc[[index]][1].values[0]):
            print(Data.loc[[index]][1].values[0],Data.loc[[index+2]][1].values[0])
            
            sumrow1 += Data.loc[[index+2]].values.flatten()
            
            sumcounter += 1
            count = index+1
            newframe[nfcount-1,:] = Data.loc[[index+2]].values.flatten()
            nfcount += 1
        else:
            count = -1
    
    newframe = newframe[0:sumcounter,:]

    DataNew = pd.DataFrame(newframe,columns=kolonner)
    
    suffix = 'daymissing'
    if '1' == str(test_day):
        suffix = 'pre'
    elif '2' == str(test_day):
        suffix = 'mid'
    elif '3' == str(test_day):
        suffix = 'post'
    elif '3CPET' == str(test_day):
        suffix = 'po3'
    elif '1CPET' == str(test_day):
        suffix = 'pr3'
    else:
        suffix = 'daymissing'
    
    # Write collected data to outoutfile
    DataNew.to_csv('MultiFitsRaw_'+patient_id+'_'+suffix+'_'+str(int(trial_id))+'.csv', index=False, sep=',')
    
    # Compute variance of estimates
    count = -1
    for index in Data.index[0:-2]:
        if index <= count: continue
        
        if np.isnan(Data.loc[[index]][1].values[0]) and np.isnan(Data.loc[[index+2]][1].values[0]):
            count = index + 1
        elif np.isnan(Data.loc[[index]][1].values[0]):
            varrow1 += (Data.loc[[index+2]].values.flatten()-sumrow1/sumcounter)**2
            
            count = index + 1
        else:
            count = -1
    
    print(sumcounter)
    
    print("Middelverdier")
    gjsnitt = sumrow1/sumcounter
    print(gjsnitt)
    
    print("Varianser")
    varianser = np.sqrt(varrow1/(sumcounter-1))
    print(varianser)
    
