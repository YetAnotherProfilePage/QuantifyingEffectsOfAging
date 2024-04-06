
import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from Models.ModelD1 import *
from Models.ModelD2 import *
from Models.ModelD3 import *
from Models.ModelD4 import *

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

Adult_ModelD1_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (1,1e7), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelD1_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (1e-16,1e-4), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

#ModelD1_Fitted_Params_Indices = [0,1,3,4,5,6,7]
ModelD1_Fitted_Params_Indices = [7,]
ModelD1_Params_Labels = ['beta','d_I','p','c','d_T','r','k_T','K']

#beta,d_I,p,c,d_T,r,k_T,c_T,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Adult_ModelD2_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (1e-4,1e0),#c_T
    (1,1e7), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelD2_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4,1e2),#r
    #(r,r),#r
    (1e1,1e8),#k_T
    (1e-4,1e0),#c_T
    (1,1e7), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

ModelD2_Fitted_Params_Indices = [0,1,3,4,5,6,7,8]
ModelD2_Params_Labels = ['beta','d_I','p','c','d_T','r','k_T','c_T','K']

#beta,d_I,p,c,d_T,r,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Adult_ModelD3_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (1,1e7), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]

#beta,d_I,p,c,d_T,r,U_0,I_0,V_0,T_0
#Param values largely taken from Esteban's paper 2014
Aged_ModelD3_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (1,1e7), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

ModelD3_Fitted_Params_Indices = [0,1,3,4,5,6]
ModelD3_Params_Labels = ['beta','d_I','p','c','d_T','r','K']

#Param values largely taken from Esteban's paper 2014
Adult_ModelD4_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0), #p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (1e-4,1e0),#c_T
    (1,1e7), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Adult,T_0_Adult)#T_0
    ]
#Param values largely taken from Esteban's paper 2014

Aged_ModelD4_Parameter_Bounds = [
    (1e-8,1e-4),#beta
    (1e-8,1e-4),#d_I
    (1e0,1e0),#p
    # Why is c fixed?
    #(4.2,4.2),#c
    (1e-2,1e2),
    (1e-4,1e0),#d_T
    (1e-4/k_V,1e2/k_V),#r
    #(r,r),#r
    (1e-4,1e0),#c_T
    (1,1e7), #K
    (1e6,1e6),#U_0
    (0.0,0.0),#I_0
    (V_0,V_0),#V_0
    (T_0_Aged,T_0_Aged)#T_0
    ]

ModelD4_Fitted_Params_Indices = [0,1,3,4,5,6,7]
ModelD4_Params_Labels = ['beta','d_I','p','c','d_T','r','c_T','K']

test_bounds = [(1,10),(0,1)]

print(generate_param_bounds(test_bounds,1,10,'standard'))

def Create_Likelihood_Profile(model_loss, samples, bounds, indices, param_labels, Viral_Data, CD8_Data,file_root,bounds_scale = 'log',repeats=3):
    dfs = []
    for i in indices:
        curr_index = param_labels[i]
        col_names = [*param_labels,'U_0','I_0','T_0','V_0','RMSLE']
        #print(col_names)
        sampled = []
        bounds_i = generate_param_bounds(bounds,i,samples,bounds_scale)
        #print(bounds_i)
        for i in range(samples):
            print(f'Param {curr_index}, Sample {i}/{samples}')
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

        print()
        print(sampled[1])
        print(col_names)

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

    #Create_Likelihood_Profile(ModelD1_RMSLE,SAMPLES,Adult_ModelD1_Parameter_Bounds,ModelD1_Fitted_Params_Indices,ModelD1_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelD1_Adult",'log',repeats=REPEATS)
    Create_Likelihood_Profile(ModelD1_RMSLE,SAMPLES,Aged_ModelD1_Parameter_Bounds,ModelD1_Fitted_Params_Indices,ModelD1_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelD1_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelD2_RMSLE,SAMPLES,Adult_ModelD2_Parameter_Bounds,ModelD2_Fitted_Params_Indices,ModelD2_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelD2_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelD2_RMSLE,SAMPLES,Aged_ModelD2_Parameter_Bounds,ModelD2_Fitted_Params_Indices,ModelD2_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelD2_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelD3_RMSLE,SAMPLES,Adult_ModelD3_Parameter_Bounds,ModelD3_Fitted_Params_Indices,ModelD3_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelD3_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelD3_RMSLE,SAMPLES,Aged_ModelD3_Parameter_Bounds,ModelD3_Fitted_Params_Indices,ModelD3_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelD3_Aged",'log',repeats=REPEATS)

    #Create_Likelihood_Profile(ModelD4_RMSLE,SAMPLES,Adult_ModelD4_Parameter_Bounds,ModelD4_Fitted_Params_Indices,ModelD4_Params_Labels,Adult_Viral_Data,Adult_CD8_Data,"ModelD4_Adult",'log',repeats=REPEATS)
    #Create_Likelihood_Profile(ModelD4_RMSLE,SAMPLES,Aged_ModelD4_Parameter_Bounds,ModelD4_Fitted_Params_Indices,ModelD4_Params_Labels,Aged_Viral_Data,Aged_CD8_Data,"ModelD4_Aged",'log',repeats=REPEATS)

    end = time.time()
    print(end-start)
    
