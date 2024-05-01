
import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from ModelBounds import *
from FittingUtilities import *

from Models.ModelC1 import *
from Models.ModelC2 import *
from Models.ModelC3 import *
from Models.ModelC4 import *


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

Adult_ModelC1_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-16,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0) #V_0
    ]

Aged_ModelC1_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-16,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0) #V_0
    ]

#ModelC1_Fitted_Params_Indices = [0,2,3,4,5,6]
ModelC1_Fitted_Params_Indices = [6,]
#Params_Labels should be all parameter labels, not just the ones fit.
ModelC1_Params_Labels = ['p','k_V','c_V','r','k_T','d_T','K']

Adult_ModelC2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (1e-16,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelC2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (1e-16,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

ModelC2_Fitted_Params_Indices = [0,2,3,4,5,6,7]
ModelC2_Params_Labels = ['p','k_V','c_V','r','k_T','d_T','c_T','K']

Adult_ModelC3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #c_T
    (1e-16,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelC3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #c_T
    (1e-16,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

ModelC3_Fitted_Params_Indices = [0,2,3,4,5]
ModelC3_Params_Labels = ['p','k_V','c_V','r','d_T','K']

Adult_ModelC4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (1e-16,1e-4), #K
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelC4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (1e-16,1e-4), #K
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]


ModelC4_Fitted_Params_Indices = [0,2,3,4,5,6]
ModelC4_Params_Labels = ['p','k_V','c_V','r','d_T','c_T','K']

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
        for i in range(samples):
            print(f'Param {curr_index}, Sample {i}/{samples}')
            print(bounds_i[i])
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
        #print(curr_index)
        #print(sampled)
        #print(col_names)
        print()
        df = pd.DataFrame(data=sampled,columns=col_names)
        print(df)
        print()

        df.to_csv(f'LikelihoodProfileFits/{file_root}-{curr_index}.csv')


    return 0

SAMPLES = 30
REPEATS = 3


if __name__ == "__main__":
    print('starting')

    start = time.time()

    Create_Likelihood_Profile(ModelC1_RMSLE,SAMPLES,Adult_ModelC1_Parameter_Bounds,ModelC1_Fitted_Params_Indices,ModelC1_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelC1_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelC1_RMSLE,SAMPLES,Aged_ModelC1_Parameter_Bounds,ModelC1_Fitted_Params_Indices,ModelC1_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelC1_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelC2_RMSLE,SAMPLES,Adult_ModelC2_Parameter_Bounds,ModelC2_Fitted_Params_Indices,ModelC2_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelC2_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelC2_RMSLE,SAMPLES,Aged_ModelC2_Parameter_Bounds,ModelC2_Fitted_Params_Indices,ModelC2_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelC2_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelC3_RMSLE,SAMPLES,Adult_ModelC3_Parameter_Bounds,ModelC3_Fitted_Params_Indices,ModelC3_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelC3_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelC3_RMSLE,SAMPLES,Aged_ModelC3_Parameter_Bounds,ModelC3_Fitted_Params_Indices,ModelC3_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelC3_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelC4_RMSLE,SAMPLES,Adult_ModelC4_Parameter_Bounds,ModelC4_Fitted_Params_Indices,ModelC4_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelC4_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelC4_RMSLE,SAMPLES,Aged_ModelC4_Parameter_Bounds,ModelC4_Fitted_Params_Indices,ModelC4_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelC4_Aged",'log',repeats=REPEATS)

    end = time.time()
    print(end-start)
    
