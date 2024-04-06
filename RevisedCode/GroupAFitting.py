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

if __name__ == "__main__":
    print('starting')
    
    #Number of threads to use. Set lower if necessary.
    WORKERS = -1

    start = time.time()

    print('Entering modelA1 fit')
    print('Adult fit')
    adult_modelA1_fit = sp.optimize.differential_evolution(func=ModelA1_RMSLE, bounds = Adult_ModelA1_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelA1_fit = sp.optimize.differential_evolution(func=ModelA1_RMSLE, bounds = Aged_ModelA1_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelA1_fit)
    print(aged_modelA1_fit)

    np.savetxt('ModelFits/adult_modelA1_params.txt', np.array([adult_modelA1_fit.x]))
    np.savetxt('ModelFits/adult_modelA1_fit.txt', np.array([adult_modelA1_fit.fun]))
    np.savetxt('ModelFits/aged_modelA1_params.txt', np.array([aged_modelA1_fit.x]))
    np.savetxt('ModelFits/aged_modelA1_fit.txt', np.array([aged_modelA1_fit.fun]))

    
    print('Entering modelA2 fit')
    print('Adult fit')
    adult_modelA2_fit = sp.optimize.differential_evolution(func=ModelA2_RMSLE, bounds = Adult_ModelA2_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelA2_fit = sp.optimize.differential_evolution(func=ModelA2_RMSLE, bounds = Aged_ModelA2_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelA2_fit)
    print(aged_modelA2_fit)

    np.savetxt('ModelFits/adult_modelA2_params.txt', np.array([adult_modelA2_fit.x]))
    np.savetxt('ModelFits/adult_modelA2_fit.txt', np.array([adult_modelA2_fit.fun]))
    np.savetxt('ModelFits/aged_modelA2_params.txt', np.array([aged_modelA2_fit.x]))
    np.savetxt('ModelFits/aged_modelA2_fit.txt', np.array([aged_modelA2_fit.fun]))
    
    print('Entering modelA3 fit')
    print('Adult fit')
    adult_modelA3_fit = sp.optimize.differential_evolution(func=ModelA3_RMSLE, bounds = Adult_ModelA3_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelA3_fit = sp.optimize.differential_evolution(func=ModelA3_RMSLE, bounds = Aged_ModelA3_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelA3_fit)
    print(aged_modelA3_fit)

    np.savetxt('ModelFits/adult_modelA3_params.txt', np.array([adult_modelA3_fit.x]))
    np.savetxt('ModelFits/adult_modelA3_fit.txt', np.array([adult_modelA3_fit.fun]))
    np.savetxt('ModelFits/aged_modelA3_params.txt', np.array([aged_modelA3_fit.x]))
    np.savetxt('ModelFits/aged_modelA3_fit.txt', np.array([aged_modelA3_fit.fun]))
    
    print('Entering modelA4 fit')
    print('Adult fit')
    adult_modelA4_fit = sp.optimize.differential_evolution(func=ModelA4_RMSLE, bounds = Adult_ModelA4_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelA4_fit = sp.optimize.differential_evolution(func=ModelA4_RMSLE, bounds = Aged_ModelA4_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelA4_fit)
    print(aged_modelA4_fit)

    np.savetxt('ModelFits/adult_modelA4_params.txt', np.array([adult_modelA4_fit.x]))
    np.savetxt('ModelFits/adult_modelA4_fit.txt', np.array([adult_modelA4_fit.fun]))
    np.savetxt('ModelFits/aged_modelA4_params.txt', np.array([aged_modelA4_fit.x]))
    np.savetxt('ModelFits/aged_modelA4_fit.txt', np.array([aged_modelA4_fit.fun]))

    print(adult_modelA1_fit.fun)
    print(aged_modelA1_fit.fun)
    print(adult_modelA2_fit.fun)
    print(aged_modelA2_fit.fun)
    print(adult_modelA3_fit.fun)
    print(aged_modelA3_fit.fun)
    print(adult_modelA4_fit.fun)
    print(aged_modelA4_fit.fun)

    end = time.time()
    print(f"Time Taken: {end-start}")
