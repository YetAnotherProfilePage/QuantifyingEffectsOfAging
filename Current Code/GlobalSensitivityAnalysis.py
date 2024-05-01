from SALib.sample import saltelli
from SALib.analyze import sobol

#from SALib.test_functions import Ishigami
import numpy as np
import pandas as pd

from ModelBounds import *
from FittingUtilities import *

SAMPLES = 32*(2**15)
#SAMPLES=8

DESTINATION_DIRECTORY = "GlobalSensitivityAnalysisResults/"

CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

"""
#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0

Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']

T_0_Adult = np.mean(Adult_CD8_Data[Adult_CD8_Data['DPI']==0]['CD8+ per g/tissue'])
T_0_Aged = np.mean(Aged_CD8_Data[Aged_CD8_Data['DPI']==0]['CD8+ per g/tissue'])

V_0 = 25.0

t = np.linspace(0,19,190)

#Note: We do not distinguish k_V between Aged and Adult.
k_V = np.max(Viral_Data['Viral Titer (Pfu/ml)']) #k_V = 1.2e6
"""

## Define function to conduct SA for given model.
def RunSobolSA(model,responses,samples = 16,additional_response_args=None):
    var_names = Models_Inputs[model]
    num_vars = len(var_names)

    response_names = []
    response_dicts = []
    for k,v in responses.items():
        response_names.append(k)
        response_dicts.append(v)

    adult_bounds = Model_Bounds[f"{model}_Adult"]
    aged_bounds = Model_Bounds[f"{model}_Aged"]

    fixed_params_indices = []
    var_params_indices = []
    for i in range(len(adult_bounds)):
        if adult_bounds[i][0] == adult_bounds[i][1]:
            fixed_params_indices.append(i)
        else:
            var_params_indices.append(i)

    used_adult_bounds = [adult_bounds[i] for i in var_params_indices]
    used_aged_bounds = [aged_bounds[i] for i in var_params_indices]
    used_var_names = [var_names[i] for i in var_params_indices]
    num_used_vars = len(used_var_names)


    # Need to reconstruct var_names and num_vars based on which inputs are fixed.
    # We also need to enable reconstruction of these values.

    adult_problem = {
        'num_vars': num_used_vars,
        'names': used_var_names,
        'bounds': used_adult_bounds
    }

    aged_problem = {
        'num_vars': num_used_vars,
        'names': used_var_names,
        'bounds': used_aged_bounds
    }

    adult_param_values = saltelli.sample(adult_problem, SAMPLES,calc_second_order=True)
    aged_param_values = saltelli.sample(aged_problem, SAMPLES,calc_second_order=True)

    for output in response_names:
        Y_adult = np.zeros([adult_param_values.shape[0]])
        Y_aged = np.zeros([aged_param_values.shape[0]])
        N = len(Y_adult)
        for i, X in enumerate(adult_param_values):
            if i%1000==0:
                print(f'{model}, {output}, Adult: {i} / {N}')
                
            X_act = np.zeros(num_vars)
            X_act[var_params_indices]=X
            X_act[fixed_params_indices]=[adult_bounds[i][0] for i in fixed_params_indices]
            Y_adult[i] = responses[output][model](X_act,*additional_response_args)

        for i, X in enumerate(aged_param_values):
            if i%1000==0:
                print(f'{model}, {output}, Aged: {i} / {N}')
                
            X_act = np.zeros(num_vars)
            X_act[var_params_indices]=X
            X_act[fixed_params_indices]=[aged_bounds[i][0] for i in fixed_params_indices]            
            Y_aged[i] = responses[output][model](X_act,*additional_response_args)

        ## Conduct SA
        Si_Adult = sobol.analyze(adult_problem, Y_adult)
        total_Si_Adult, first_Si_Adult, second_Si_Adult = Si_Adult.to_df()

        Si_Aged = sobol.analyze(aged_problem, Y_aged)
        total_Si_Aged, first_Si_Aged, second_Si_Aged = Si_Aged.to_df()

        ## Save results to file.

        total_Si_Adult.to_csv(f"{DESTINATION_DIRECTORY}Adult_{model}_{output}_Total_Indices.csv")
        first_Si_Adult.to_csv(f"{DESTINATION_DIRECTORY}Adult_{model}_{output}_First_Indices.csv")
        second_Si_Adult.to_csv(f"{DESTINATION_DIRECTORY}Adult_{model}_{output}_Second_Indices.csv")

        total_Si_Aged.to_csv(f"{DESTINATION_DIRECTORY}Aged_{model}_{output}_Total_Indices.csv")
        first_Si_Aged.to_csv(f"{DESTINATION_DIRECTORY}Aged_{model}_{output}_First_Indices.csv")
        second_Si_Aged.to_csv(f"{DESTINATION_DIRECTORY}Aged_{model}_{output}_Second_Indices.csv")    

if __name__ == "__main__":
    #for model in Models:
    #    RunSobolSA(model,responses = {"ViralClearanceTime":Models_ViralClearanceTime_Dict,"TotalViralLoad":Models_TotalViralLoad_Dict,"ExcessCTL":Models_ExcessCTL_Dict}, samples = SAMPLES,additional_response_args = (19,1000))
    RunSobolSA("MD1",responses = {"ViralClearanceTime":Models_ViralClearanceTime_Dict,"TotalViralLoad":Models_TotalViralLoad_Dict,"ExcessCTL":Models_ExcessCTL_Dict}, samples = SAMPLES,additional_response_args = (19,1000))


