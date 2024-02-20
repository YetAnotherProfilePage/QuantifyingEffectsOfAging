import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from Models.ModelC1 import *
from Models.ModelD1 import *

from ModelBounds_SharedInitialConditions import *

plt.set_cmap('tab20')

POPSIZE=100
MAXITER=500

TOL = 1e-16

#TODO: Some of the following boilerplate can be cleaned up.
CD8_Data = pd.read_csv("InitialConditionsTest/ExperimentalData_SharedInitialT/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("InitialConditionsTest/ExperimentalData_SharedInitialT/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0

Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']

T_0_Aged = np.mean(Aged_CD8_Data[Aged_CD8_Data['DPI']==0]['CD8+ per g/tissue'])
T_0_Adult = T_0_Aged

t = np.linspace(0,19,190)

#Note: We do not distinguish k_V between Aged and Adult.
k_V = np.max(Viral_Data['Viral Titer (Pfu/ml)']) #k_V = 1.2e6

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

if __name__ == "__main__":
    print('starting')
    
    #Number of threads to use. Set lower if necessary.
    WORKERS = -1

    start = time.time()
        
    print('Entering Adult modelC1 fit')
    print('Adult fit')
    adult_modelC1_fit = sp.optimize.differential_evolution(func=ModelC1_RMSLE, bounds = Adult_ModelC1_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelC1_fit)
    
    np.savetxt('InitialConditionsTest/ModelFits/adult_modelC1_params.txt', np.array([adult_modelC1_fit.x]))
    
    print('Entering Aged modelD1 fit')
    aged_modelD1_fit = sp.optimize.differential_evolution(func=ModelD1_RMSLE, bounds = Aged_ModelD1_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(aged_modelD1_fit)

    np.savetxt('InitialConditionsTest/ModelFits/aged_modelD1_params.txt', np.array([aged_modelD1_fit.x]))
    end = time.time()
    print(f"Time taken: {end-start}")