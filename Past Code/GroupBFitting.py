import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from Models.ModelB1 import *
from Models.ModelB2 import *
from Models.ModelB3 import *
from Models.ModelB4 import *

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

    print('Entering modelB1 fit')
    print('Adult fit')
    adult_modelB1_fit = sp.optimize.differential_evolution(func=ModelB1_RMSLE, bounds = Adult_ModelB1_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    #aged_modelB1_fit = sp.optimize.differential_evolution(func=ModelB1_RMSLE, bounds = Aged_ModelB1_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelB1_fit)
    #print(aged_modelB1_fit)

    np.savetxt('ModelFits/adult_modelB1_params.txt', np.array([adult_modelB1_fit.x]))
    np.savetxt('ModelFits/adult_modelB1_fit.txt', np.array([adult_modelB1_fit.fun]))
    #np.savetxt('ModelFits/aged_modelB1_params.txt', np.array([aged_modelB1_fit.x]))
    #np.savetxt('ModelFits/aged_modelB1_fit.txt', np.array([aged_modelB1_fit.fun]))
    
    print('Entering modelB2 fit')
    print('Adult fit')
    adult_modelB2_fit = sp.optimize.differential_evolution(func=ModelB2_RMSLE, bounds = Adult_ModelB2_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    #aged_modelB2_fit = sp.optimize.differential_evolution(func=ModelB2_RMSLE, bounds = Aged_ModelB2_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelB2_fit)
    #print(aged_modelB2_fit)

    np.savetxt('ModelFits/adult_modelB2_params.txt', np.array([adult_modelB2_fit.x]))
    np.savetxt('ModelFits/adult_modelB2_fit.txt', np.array([adult_modelB2_fit.fun]))
    #np.savetxt('ModelFits/aged_modelB2_params.txt', np.array([aged_modelB2_fit.x]))
    #np.savetxt('ModelFits/aged_modelB2_fit.txt', np.array([aged_modelB2_fit.fun]))

    
    print('Entering modelA3 fit')
    print('Adult fit')
    adult_modelB3_fit = sp.optimize.differential_evolution(func=ModelB3_RMSLE, bounds = Adult_ModelB3_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    #aged_modelB3_fit = sp.optimize.differential_evolution(func=ModelB3_RMSLE, bounds = Aged_ModelB3_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelB3_fit)
    #print(aged_modelB3_fit)

    np.savetxt('ModelFits/adult_modelB3_params.txt', np.array([adult_modelB3_fit.x]))
    np.savetxt('ModelFits/adult_modelB3_fit.txt', np.array([adult_modelB3_fit.fun]))
    #np.savetxt('ModelFits/aged_modelB3_params.txt', np.array([aged_modelB3_fit.x]))
    #np.savetxt('ModelFits/aged_modelB3_fit.txt', np.array([aged_modelB3_fit.fun]))

    

    print('Entering modelB4 fit')
    print('Adult fit')
    adult_modelB4_fit = sp.optimize.differential_evolution(func=ModelB4_RMSLE, bounds = Adult_ModelB4_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    #aged_modelB4_fit = sp.optimize.differential_evolution(func=ModelB4_RMSLE, bounds = Aged_ModelB4_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_modelB4_fit)
    #print(aged_modelB4_fit)

    np.savetxt('ModelFits/adult_modelB4_params.txt', np.array([adult_modelB4_fit.x]))
    np.savetxt('ModelFits/adult_modelB4_fit.txt', np.array([adult_modelB4_fit.fun]))
    #np.savetxt('ModelFits/aged_modelB4_params.txt', np.array([aged_modelB4_fit.x]))
    #np.savetxt('ModelFits/aged_modelB4_fit.txt', np.array([aged_modelB4_fit.fun]))


    print(adult_modelB1_fit.fun)
    print(aged_modelB1_fit.fun)
    print(adult_modelB2_fit.fun)
    print(aged_modelB2_fit.fun)
    print(adult_modelB3_fit.fun)
    print(aged_modelB3_fit.fun)
    print(adult_modelB4_fit.fun)
    print(aged_modelB4_fit.fun)

    
    end = time.time()
    print(f"Time Taken: {end-start}")
