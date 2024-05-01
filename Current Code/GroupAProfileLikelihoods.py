
import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from Models.ModelA1 import *
from Models.ModelA2 import *
from Models.ModelA3 import *
from Models.ModelA4 import *
from ModelBounds import *

plt.set_cmap('tab20')

POPSIZE=100
MAXITER=500


#Number of threads to use. Set lower if necessary.
WORKERS = -1


TOL = 1e-16

#TODO: Some of the following boilerplate can be cleaned up.
CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0


Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']

T_0_Adult = np.mean(Adult_CD8_Data[Adult_CD8_Data['DPI']==0]['CD8+ per g/tissue'])
T_0_Aged = np.mean(Aged_CD8_Data[Aged_CD8_Data['DPI']==0]['CD8+ per g/tissue'])

print(T_0_Adult)

t = np.linspace(0,19,190)

#Note: We do not distinguish k_V between Aged and Adult.
k_V = np.max(Viral_Data['Viral Titer (Pfu/ml)']) #k_V = 1.2e6

#print(np.max(CD8_Data['CD8+ per g/tissue'])) #Order 1e7

V_0 = 25.0
d_T = 0.02
p = 4.4
#c_V = 2.5e-6
c_V = 2.61e-6
#r = 0.33
r=0.20
k_T = 2.7e3


def trim_to_threshhold(data, threshold = 25.0, minimum=True):
    #Utility to censor data at threshold.
    if minimum==True:
        return np.where(data<threshold,threshold,data)
    else:
        return np.where(data>threshold,threshold,data)

def generate_param_bounds(bounds, param_idx_of_interest, num_samples = 30,scale='standard'):
    res = []
    #print(bounds[param_idx_of_interest])
    #print(*bounds[param_idx_of_interest])
    if scale =='standard':
        test_vals = np.linspace(bounds[param_idx_of_interest][0],bounds[param_idx_of_interest][1],num_samples)
    elif scale == 'log':
        test_vals = np.linspace(np.log(bounds[param_idx_of_interest][0]),np.log(bounds[param_idx_of_interest][1]),num_samples)
    #print(test_vals)
    for i in range(num_samples):
        new_bounds = bounds.copy()
        if scale=='log':
            new_bounds[param_idx_of_interest]=(np.exp(test_vals[i]),np.exp(test_vals[i]))
        else:
            new_bounds[param_idx_of_interest]=(test_vals[i],test_vals[i])
        res.append(new_bounds)
    return res



# Note that in these parameter bounds, c_T and d_T are reverse labelled. It's an inconsistency I should take the time to fix at some point.

ModelA1_Fitted_Params_Indices = [0,2,3,4]
ModelA1_Params_Labels = ['p','k_V','c_V','r','k_T','d_T']

ModelA2_Fitted_Params_Indices = [0,2,3,4,6]
ModelA2_Params_Labels = ['p','k_V','c_V','r','k_T','d_T','c_T']

ModelA3_Fitted_Params_Indices = [0,2,3,4,5]
ModelA3_Params_Labels = ['p','k_V','c_V','r','k_T','d_T']


ModelA4_Fitted_Params_Indices = [0,2,3,4,5,6]
ModelA4_Params_Labels = ['p','k_V','c_V','r','k_T','d_T','c_T']


test_bounds = [(1,10),(0,1)]

print(generate_param_bounds(test_bounds,1,10,'standard'))


def Create_Likelihood_Profile(model_loss, samples, bounds, indices, param_labels, Viral_Data, CD8_Data,file_root,bounds_scale = 'log',repeats=3):
    dfs = []
    for i in indices:
        curr_index = param_labels[i]
        col_names = [*param_labels,'T_0','V_0','RMSLE']
        #print(col_names)
        sampled = []
        bounds_i = generate_param_bounds(bounds,i,samples,bounds_scale)
        #print(bounds_i)
        print(curr_index)
        for i in range(samples):
            print(f'Sample {i}/{samples}')
            if repeats > 1:
                fits = []
                scores = []
                for j in range(repeats):
                    print(f'In param {curr_index}, In repeat {j+1}/{repeats}')
                    this_fit = sp.optimize.differential_evolution(func=model_loss, bounds = bounds_i[i],args=(19,1000,Viral_Data,CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
                    fits.append(this_fit)
                    scores.append(this_fit.fun)

                fit = fits[np.argmin(scores)]

            else:
                fit = sp.optimize.differential_evolution(func=model_loss, bounds = bounds_i[i],args=(19,1000,Viral_Data,CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
            sampled.append([*fit.x,fit.fun])
        #print()
        print(curr_index)
        #print(sampled)
        #print(col_names)
        print()
        df = pd.DataFrame(data=sampled,columns=col_names)
        print(df)
        print()

        df.to_csv(f'LikelihoodProfileFits/{file_root}-{curr_index}.csv')


    return 0

SAMPLES = 10
REPEATS = 1


if __name__ == "__main__":
    print('starting')

    start = time.time()

    #Create_Likelihood_Profile(ModelA1_RMSLE,SAMPLES,Adult_ModelA1_Parameter_Bounds,ModelA1_Fitted_Params_Indices,ModelA1_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelA1_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelA1_RMSLE,SAMPLES,Aged_ModelA1_Parameter_Bounds,ModelA1_Fitted_Params_Indices,ModelA1_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelA1_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelA2_RMSLE,SAMPLES,Adult_ModelA2_Parameter_Bounds,ModelA2_Fitted_Params_Indices,ModelA2_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelA2_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelA2_RMSLE,SAMPLES,Aged_ModelA2_Parameter_Bounds,ModelA2_Fitted_Params_Indices,ModelA2_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelA2_Aged",'log',repeats=REPEATS)

    Create_Likelihood_Profile(ModelA3_RMSLE,SAMPLES,Adult_ModelA3_Parameter_Bounds,ModelA3_Fitted_Params_Indices,ModelA3_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelA3_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelA3_RMSLE,SAMPLES,Aged_ModelA3_Parameter_Bounds,ModelA3_Fitted_Params_Indices,ModelA3_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelA3_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelA4_RMSLE,SAMPLES,Adult_ModelA4_Parameter_Bounds,ModelA4_Fitted_Params_Indices,ModelA4_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelA4_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelA4_RMSLE,SAMPLES,Aged_ModelA4_Parameter_Bounds,ModelA4_Fitted_Params_Indices,ModelA4_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelA4_Aged",'log',repeats=REPEATS)

    end = time.time()
    print(end-start)
    
