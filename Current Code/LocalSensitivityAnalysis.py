import numpy as np
import pandas as pd
import itertools
import time

from ModelBounds import *
from FittingUtilities import *

#SAMPLES = 32*1024
SAMPLES=100

DESTINATION_DIRECTORY = "LocalSensitivityAnalysisResults/"
MODEL_FITS_FOLDER = "ModelFits/"

groups = ['adult','aged']
combs = list(itertools.product(groups,Models))
file_names = []
model_estimates = {}
for comb in combs:
    file_name= f"{MODEL_FITS_FOLDER}{comb[0]}_model{comb[1][1:]}_params.txt"
    model_estimates[f"{comb[1]}_{comb[0]}"]=np.loadtxt(file_name)

def Create_Param_Samples(bounds, param_idx, samples):
    #need to sample in log.
    lower, upper = bounds[param_idx]
    #print(lower)
    #print(upper)
    #This prevents numerical errors
    if lower == 0:
        lower = 1e-16
    vals = np.linspace(np.log10(lower),np.log10(upper),samples)
    #print(vals)
    return np.power(10.0,vals)

def Run_OAT_SA(model,group,estimate,responses,SAMPLE_SIZE = 16,additional_response_args=None):
    #estimate will take in the estimate of the model.
    var_names = Models_Inputs[model]
    #print(var_names)
    num_vars = len(var_names)

    response_names = []
    response_dicts = []
    for k,v in responses.items():
        response_names.append(k)
        response_dicts.append(v)

    #print(response_names)
    #print(response_dicts)
    #print(responses)
    #print(response_dicts)

    adult_bounds = Model_Bounds[f"{model}_Adult"]
    aged_bounds = Model_Bounds[f"{model}_Aged"]

    fixed_params_indices = []
    var_params_indices = []
    for i in range(len(adult_bounds)):
        if adult_bounds[i][0] == adult_bounds[i][1]:
            print(adult_bounds[i][0])
            fixed_params_indices.append(i)
        else:
            var_params_indices.append(i)

    used_adult_bounds = [adult_bounds[i] for i in var_params_indices]
    used_aged_bounds = [aged_bounds[i] for i in var_params_indices]
    used_var_names = [var_names[i] for i in var_params_indices]
    num_used_vars = len(used_var_names)

    #print(used_var_names)
    #print()


    # Need to reconstruct var_names and num_vars based on which inputs are fixed.
    # We also need to enable reconstruction of these values.

    for param_idx in var_params_indices:
        # Get sample points across range.
        sample_points = Create_Param_Samples(adult_bounds,param_idx,SAMPLE_SIZE)
        temp = estimate.copy()
        samples = []

        print(param_idx)
        print(var_names[param_idx])

        for point in sample_points:
            new = temp.copy()
            new[param_idx]=point
            samples.append(new)

        #if var_names[param_idx]=="K":
        #    print(sample_points)

        # Evaluate outcomes across the samples
        names = [var_names[param_idx], *response_names]
        #print(names)
        #print(var_names[param_idx])
        responses = []

        #print(sample_points)
        
        for sample in samples:
            response = [sample[param_idx],]
            for idx in range(len(response_names)):
                response.append(response_dicts[idx][model](sample,19,10))
            responses.append(response)

        df = pd.DataFrame(data=responses,columns=names)
        #print(df)
        df.to_csv(f"{DESTINATION_DIRECTORY}{model}-{group}-{var_names[param_idx]}-LocalSensitivity.csv")

        ## Conduct SA
        ## Save results to file. of form {model}_{group}_{param}_localsensitivity.py

        
if __name__ == "__main__":
    #Create_Param_Samples(Model_Bounds["MC1_Adult"],6,100)

    Run_OAT_SA("MA3","adult",model_estimates["MA3_adult"],{'Viral Clearance Time':Models_ViralClearanceTime_Dict,'Total Viral Load':Models_TotalViralLoad_Dict,'Excess CTL':Models_ExcessCTL_Dict},SAMPLE_SIZE=SAMPLES)
    Run_OAT_SA("MB1","aged",model_estimates["MB1_aged"],{'Viral Clearance Time':Models_ViralClearanceTime_Dict,'Total Viral Load':Models_TotalViralLoad_Dict,'Excess CTL':Models_ExcessCTL_Dict},SAMPLE_SIZE=SAMPLES)
    """
    for model in Models:
        print(f"{model}, adult")
        start = time.time()
        Run_OAT_SA(model,"adult",model_estimates[f"{model}_adult"],{'Viral Clearance Time':Models_ViralClearanceTime_Dict,'Total Viral Load':Models_TotalViralLoad_Dict,'Excess CTL':Models_ExcessCTL_Dict},SAMPLE_SIZE=SAMPLES)
        end = time.time()
        print(f"Time taken: {end-start}")

        print(f"{model}, aged")
        start = time.time()
        Run_OAT_SA(model,"aged",model_estimates[f"{model}_aged"],{'Viral Clearance Time':Models_ViralClearanceTime_Dict,'Total Viral Load':Models_TotalViralLoad_Dict,'Excess CTL':Models_ExcessCTL_Dict},SAMPLE_SIZE=SAMPLES)
        end = time.time()
        print(f"Time taken: {end-start}")

        pass
        #RunSobolSA(model,responses = {"ViralClearanceTime":Models_ViralClearanceTime_Dict,"TotalViralLoad":Models_TotalViralLoad_Dict,"ExcessCTL":Models_ExcessCTL_Dict}, samples = SAMPLES,additional_response_args = (19,1000))
    """
    
