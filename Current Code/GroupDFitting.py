import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from Models.ModelD1 import *
from Models.ModelD2 import *
from Models.ModelD3 import *
from Models.ModelD4 import *

from ModelBounds import *


plt.set_cmap('tab20')

POPSIZE=3
MAXITER=3

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
    
    # Note that here we attempt more traditional gradient based methods, but see that they perform worse.
    #test = sp.optimize.minimize(fun=ModelA1_RMSLE, x0 = [1, k_V, 1e-6, 1e-1, 1e5, 1e-2,T_0_Adult,V_0], method="SLSQP", bounds = Aged_ModelA1_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data))
    #print(test)

    print('Entering modelD1 fit')
    print('Adult fit')
    adult_modelD1_fit = sp.optimize.differential_evolution(func=ModelD1_RMSLE, bounds = Adult_ModelD1_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelD1_fit = sp.optimize.differential_evolution(func=ModelD1_RMSLE, bounds = Aged_ModelD1_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelD1_fit)
    print(aged_modelD1_fit)

    np.savetxt('ModelFits/adult_modelD1_params.txt', np.array([adult_modelD1_fit.x]))
    np.savetxt('ModelFits/adult_modelD1_fit.txt', np.array([adult_modelD1_fit.fun]))
    np.savetxt('ModelFits/aged_modelD1_params.txt', np.array([aged_modelD1_fit.x]))
    np.savetxt('ModelFits/aged_modelD1_fit.txt', np.array([aged_modelD1_fit.fun]))
    
    
    print('Entering modelD2 fit')
    print('Adult fit')
    adult_modelD2_fit = sp.optimize.differential_evolution(func=ModelD2_RMSLE, bounds = Adult_ModelD2_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelD2_fit = sp.optimize.differential_evolution(func=ModelD2_RMSLE, bounds = Aged_ModelD2_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelD2_fit)
    print(aged_modelD2_fit)

    np.savetxt('ModelFits/adult_modelD2_params.txt', np.array([adult_modelD2_fit.x]))
    np.savetxt('ModelFits/adult_modelD2_fit.txt', np.array([adult_modelD2_fit.fun]))
    np.savetxt('ModelFits/aged_modelD2_params.txt', np.array([aged_modelD2_fit.x]))
    np.savetxt('ModelFits/aged_modelD2_fit.txt', np.array([aged_modelD2_fit.fun]))

    
    print('Entering modelD3 fit')
    print('Adult fit')
    adult_modelD3_fit = sp.optimize.differential_evolution(func=ModelD3_RMSLE, bounds = Adult_ModelD3_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelD3_fit = sp.optimize.differential_evolution(func=ModelD3_RMSLE, bounds = Aged_ModelD3_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelD3_fit)
    print(aged_modelD3_fit)

    np.savetxt('ModelFits/adult_modelD3_params.txt', np.array([adult_modelD3_fit.x]))
    np.savetxt('ModelFits/adult_modelD3_fit.txt', np.array([adult_modelD3_fit.fun]))
    np.savetxt('ModelFits/aged_modelD3_params.txt', np.array([aged_modelD3_fit.x]))
    np.savetxt('ModelFits/aged_modelD3_fit.txt', np.array([aged_modelD3_fit.fun]))

    print('Entering modelD4 fit')
    print('Adult fit')
    adult_modelD4_fit = sp.optimize.differential_evolution(func=ModelD4_RMSLE, bounds = Adult_ModelD4_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_modelD4_fit = sp.optimize.differential_evolution(func=ModelD4_RMSLE, bounds = Aged_ModelD4_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelD4_fit)
    print(aged_modelD4_fit)

    np.savetxt('ModelFits/adult_modelD4_params.txt', np.array([adult_modelD4_fit.x]))
    np.savetxt('ModelFits/adult_modelD4_fit.txt', np.array([adult_modelD4_fit.fun]))
    np.savetxt('ModelFits/aged_modelD4_params.txt', np.array([aged_modelD4_fit.x]))
    np.savetxt('ModelFits/aged_modelD4_fit.txt', np.array([aged_modelD4_fit.fun]))
    
    end = time.time()
    print(f'Time taken: {end-start}')
