#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Remember, we fix the parameters in order to improve identifiability. To an extent, these values are arbitrary.
'''

'''
Adult Model 3 AICc: nan
Aged Model 3 AICc: 33.227013584218255
Adult Model 4 AICc: 415.20125415996716
Aged Model 4 AICc: 347.7222802192328
Adult Model 5 AICc: -46.50899416645368
Aged Model 5 AICc: 1.2826443725055992

adult_model5_params: [3.34535764e-06 8.27883190e-07 9.99999993e+00 4.20000000e+00
 2.00000000e-02 2.00000000e-01 1.49065756e+03 4.96911570e-02
 1.00000000e+06 0.00000000e+00 2.50000000e+01 9.46546400e+05]

aged_model5_params: [5.37339003e-05 5.24750207e-07 3.64504353e-01 4.20000000e+00
 2.00000000e-02 2.00000000e-01 2.39633054e-01 2.01245483e-02
 1.00000000e+06 0.00000000e+00 2.50000000e+01 4.77428000e+05]


Good parameters for aged_model3 (takes a while to run.)
[2.90782457e+00 1.20000000e+06 1.72227285e-06 6.60547183e-01
 3.45546138e+04 2.00000000e-01 5.06758118e-02 2.00000000e-02
 1.52816119e-01 4.77428000e+05 2.50000000e+01]
Aged Model 3 AICc: 33.227013584218255
'''

import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

#import Model1
#from Model1 import *
#import Models.Model1
from Models.Model1 import *

#print(model1)

#import Model3
#from Model2 import *
#import Models.Model2
from Models.Model2 import *

#import Model4
#from Model3 import *
#import Models.Model3
from Models.Model3 import *

#import Model5
#from Model4 import *
#import Models.Model4
from Models.Model4 import *

#import model6
#from Model5 import *
#import Models.Model5
from Models.Model5 import *

'''
For restricted DPI on CD8+ data.

Adult Model 1 AICc: 16.067068232410993
Aged Model 1 AICc: 34.629883125962195
Adult Model 3 AICc: 18.25195076601313
Aged Model 3 AICc: 36.79564368634584
Adult Model 4 AICc: 70.56768751130787
Aged Model 4 AICc: 53.68833539794013
Adult Model 5 AICc: 36.60551553714343
Aged Model 5 AICc: 46.63110400447426
Adult Model 6 AICc: 77.83191337342323
Aged Model 6 AICc: 77.48014226935271
Adult Model 7 AICc: 57.67932090358132
Aged Model 7 AICc: 64.29969449931318
Adult Model 8 AICc: 67.35054823310409
Aged Model 8 AICc: 68.37918967678003
'''

np.random.seed(seed=233423)

plt.set_cmap('tab20')

#POPSIZE = 30
#MAXITER = 100

POPSIZE=100
#MAXITER=100

MAXITER=500

#POPSIZE=10
#MAXITER=10

#POPSIZE=5
#MAXITER=5
TOL = 1e-16

#TODO: Some of the following boilerplate can be cleaned up.
CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-4).copy()
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


if __name__ == "__main__":
    print('starting')

    """
    Adult_Model2_Parameter_Bounds = (
        (1,8), #p
        (k_V,k_V), #k_V #1.2e6
        (5e-8,1e-5), #c_V
        #(r,r), #r
        (0,1e2),
        (0,3e8), #k_T
        (d_T,d_T), #d_T
        #(1e-10,1e4), #c_T
        (0,1e4), #c_T
        (T_0_Adult,T_0_Adult),
        (V_0,V_0), #V_0
        )

    Aged_Model2_Parameter_Bounds = (
        (1,8), #p
        (k_V,k_V), #k_V #1.2e6
        (5e-8,1e-5), #c_V
        #(r,r), #r
        (0,1e2),
        (0,3e8), #k_T
        (d_T,d_T), #d_T
        #(1e-10,1e4), #c_T
        (0,1e4), #c_T
        (T_0_Aged,T_0_Aged),
        (V_0,V_0), #V_0
        )
    
    Adult_Model1_Parameter_Bounds = (
        (1,8), #p
        (k_V,k_V), #k_V #1.2e6
        (5e-8,1e-5), #c_V
        #(r,r), #r
        (0,1e2),
        (1e-2,3e8),
        (d_T,d_T), #d_T
        (T_0_Adult,T_0_Adult),
        (V_0,V_0) #V_0
        )
    
    Aged_Model1_Parameter_Bounds = (
        (1,8), #p
        (k_V,k_V), #k_V #1.2e6
        (5e-8,1e-5), #c_V
        #(r,r), #r
        (0,1e2),
        (1e-2,3e8), #taken from Boianelli
        (d_T,d_T), #d_T
        (T_0_Aged,T_0_Aged),
        (V_0,V_0) #V_0
        )
    """

    """
    Adult_Model3_Parameter_Bounds = (
        (0.01,20), #p
        (k_V,k_V), #k_V #1.2e6
        (1e-9,1e-5), #c_V
        (1,1e8),#s_N 
        (1e-6,100000),#gamma,
        #(1,k_V),#k_N,
        (0,100000),#d_N,
        #(r,r), #r
        (0,10), #k_T
        (d_T,d_T), #d_T
        (1e-10,1), #c_T
        (T_0_Adult,T_0_Adult),
        (V_0,V_0) #V_0
        )
    
    Aged_Model3_Parameter_Bounds = (
        (0.01,20), #p
        (k_V,k_V), #k_V #1.2e6
        #(1e-9,1), #c_V
        (1e-9,1e-5), #c_V
        (1,1e8),#s_N
        (1e-6,1),#gamma,
        #(1,k_V),#k_N,
        (0,100000),#d_N,
        #(r,r), #r
        (0,10), #k_T
        (d_T,d_T), #d_T
        (1e-10,1), #c_T
        (T_0_Aged,T_0_Aged),
        (V_0,V_0) #V_0
        )
    """
    """
    Adult_Model3_Parameter_Bounds = (
        (0.01,20), #p
        (k_V,k_V), #k_V #1.2e6
        (1e-9,1e-5), #c_V
        #(1,1e8),#s_N 
        (1e-6,100000),#gamma,
        #(1,k_V),#k_N,
        (0,100000),#d_N,
        (r,r), #r
        (0,10), #k_T
        (d_T,d_T), #d_T
        (1e-10,1), #c_T
        (T_0_Adult,T_0_Adult),
        (V_0,V_0) #V_0
        )
    
    Aged_Model3_Parameter_Bounds = (
        (0.01,20), #p
        (k_V,k_V), #k_V #1.2e6
        #(1e-9,1), #c_V
        (1e-9,1e-5), #c_V
        #(1,1e8),#s_N 
        (1e-3,1),#gamma,
        #(1,k_V),#k_N,
        #(0,100000),#d_N,
        (1e3,1e5),#d_N,
        (r,r), #r
        (0,10), #k_T
        (d_T,d_T), #d_T
        (1e-10,1), #c_T
        (T_0_Aged,T_0_Aged),
        (V_0,V_0) #V_0
        )
    """
    Adult_Model3_Parameter_Bounds = (
        (0.01,20), #p
        (k_V,k_V), #k_V #1.2e6
        (1e-9,1e-3), #c_V
        #(1,1e8),#s_N 
        (1e-6,100000),#gamma,
        #(1,k_V),#k_N,
        (1e3,1e6),#d_N,
        #(r,r), #r
        (0,1e2),
        (0,10), #k_T
        (d_T,d_T), #d_T
        (1e-10,1), #c_T
        (T_0_Adult,T_0_Adult),
        (V_0,V_0) #V_0
        )
    
    Aged_Model3_Parameter_Bounds = (
        (0.01,20), #p
        (k_V,k_V), #k_V #1.2e6
        #(1e-9,1), #c_V
        (1e-9,1e-3), #c_V
        #(1,1e8),#s_N 
        (1e-3,1),#gamma,
        #(1,k_V),#k_N,
        #(0,100000),#d_N,
        (1e3,1e5),#d_N,
        #(r,r), #r
        (0,1e2),
        (0,10), #k_T
        (d_T,d_T), #d_T
        (1e-10,1), #c_T
        (T_0_Aged,T_0_Aged),
        (V_0,V_0) #V_0
        )
    
    '''
    Good parameters for aged_model3 (takes a while to run.)
    [2.90782457e+00 1.20000000e+06 1.72227285e-06 6.60547183e-01
     3.45546138e+04 2.00000000e-01 5.06758118e-02 2.00000000e-02
     1.52816119e-01 4.77428000e+05 2.50000000e+01]
    Aged Model 3 AICc: 33.227013584218255
    '''
    Adult_Model4_Parameter_Bounds = (
        (1e-5,20), #p
        (k_V,k_V), #k_V #1.2e6
        (5e-8,1e-5), #c_V
        #(r,r), #r
        (1e-16,1e-1),
        #(r*1.0/k_V,r*1.0/k_V),
        #(r*1e-10,r*1e-10),
        (d_T,d_T), #d_T
        (1e-10,1e-1), #c_T
        (T_0_Adult,T_0_Adult),
        (V_0,V_0), #V_0
        )
    
    Aged_Model4_Parameter_Bounds = (
        (1e-5,20), #p
        (k_V,k_V), #k_V #1.2e6
        (5e-8,1e-5), #c_V
        #(r,r), #r
        (1e-16,1e-1),
        #(r*1e-12,r*1e-12),
        (d_T,d_T), #d_T
        (1e-16,1e-1), #c_T
        (T_0_Aged,T_0_Aged),
        (V_0,V_0), #V_0
        )
    
    #beta,d_I,p,c,d_T,r,k_T,c_T,U_0,I_0,V_0,T_0
    #Param values largely taken from Esteban's paper 2014
    Adult_Model5_Parameter_Bounds = (
        (1e-8,1e1),#beta
        (1e-8,1e1),#d_I
        (1e-6,1e1),#p
        # Why is c fixed?
        #(4.2,4.2),#c
        (1e-1,1e2),
        (d_T,d_T),#d_T
        (1e-16,1e2),#r
        #(r,r),#r
        (1e-2,3e10),#k_T
        (1e-10,1),#c_T
        (1e6,1e6),#U_0
        (0.0,0.0),#I_0
        (V_0,V_0),#V_0
        (T_0_Adult,T_0_Adult)#T_0
        )
    #Param values largely taken from Esteban's paper 2014

    Aged_Model5_Parameter_Bounds = (
        (1e-8,1e1),#beta
        (1e-8,1e1),#d_I
        (1e-6,1e1),#p
        #(4.2,4.2),#c
        (1e-1,1e2),
        (d_T,d_T),#d_T
        (1e-16,1e2),#r
        #(r,r),#r
        (1e-2,3e10),#k_T
        (1e-10,1),#c_T
        (1e6,1e6),#U_0
        (0.0,0.0),#I_0
        (V_0,V_0),#V_0
        (T_0_Aged,T_0_Aged)#T_0
        )    
    
    def trim_to_threshhold(data, threshold = 25.0, minimum=True):
        #Utility to censor data at threshold.
        if minimum==True:
            return np.where(data<threshold,threshold,data)
        else:
            return np.where(data>threshold,threshold,data)
    
    #Number of threads to use. Set lower if necessary.
    WORKERS = 8

    start = time.time()
    
    #print(len(Adult_Model1_Parameter_Bounds))
    print('Entering model1 fit')
    print('Adult fit')
    adult_model1_fit = sp.optimize.differential_evolution(func=Model1_RMSLE, bounds = Adult_Model1_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_model1_fit = sp.optimize.differential_evolution(func=Model1_RMSLE, bounds = Aged_Model1_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_model1_fit)
    print(aged_model1_fit)

    np.savetxt('ModelFits/adult_model1_error.txt', np.array([adult_model1_fit.x]))
    np.savetxt('ModelFits/adult_model1_fit.txt', np.array([adult_model1_fit.fun]))
    np.savetxt('ModelFits/aged_model1_error.txt', np.array([aged_model1_fit.x]))
    np.savetxt('ModelFits/aged_model1_fit.txt', np.array([aged_model1_fit.fun]))

    print('Entering model2 fit')
    print('Adult fit')
    adult_model2_fit = sp.optimize.differential_evolution(func=Model2_RMSLE, bounds = Adult_Model2_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_model2_fit = sp.optimize.differential_evolution(func=Model2_RMSLE, bounds = Aged_Model2_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_model2_fit)
    print(aged_model2_fit)

    np.savetxt('ModelFits/adult_model2_error.txt', np.array([adult_model2_fit.x]))
    np.savetxt('ModelFits/adult_model2_fit.txt', np.array([adult_model2_fit.fun]))
    np.savetxt('ModelFits/aged_model2_error.txt', np.array([aged_model2_fit.x]))
    np.savetxt('ModelFits/aged_model2_fit.txt', np.array([aged_model2_fit.fun]))

    # Model 3 fit takes around 20 minutes
    """
    print('Entering model3 fit')
    print('Adult fit')
    adult_model3_fit = sp.optimize.differential_evolution(func=Model3_RMSLE, bounds = Adult_Model3_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_model3_fit = sp.optimize.differential_evolution(func=Model3_RMSLE, bounds = Aged_Model3_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(aged_model3_fit)
    print(adult_model3_fit)

    print(adult_model3_fit.fun)
    print(adult_model3_fit.x)
    np.savetxt('ModelFits/adult_model3_error.txt', np.array([adult_model3_fit.x]))
    np.savetxt('ModelFits/adult_model3_fit.txt', np.array([adult_model3_fit.fun]))
    np.savetxt('ModelFits/aged_model3_error.txt', np.array([aged_model3_fit.x]))
    np.savetxt('ModelFits/aged_model3_fit.txt', np.array([aged_model3_fit.fun]))
    """

    #res = sp.optimize.minimize(fun = Model3_RMSLE,x0=aged_model3_fit.x,args = (19,1000,Aged_Viral_Data,Aged_CD8_Data),bounds = Aged_Model3_Parameter_Bounds,tol=TOL,method='Nelder-Mead',options={'maxiter':10000})
    #print(res)
    #This one takes about 4 minutes

    print('Entering model4 fit')
    print('Adult fit')
    adult_model4_fit = sp.optimize.differential_evolution(func=Model4_RMSLE, bounds = Adult_Model4_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print('Aged fit')
    aged_model4_fit = sp.optimize.differential_evolution(func=Model4_RMSLE, bounds = Aged_Model4_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_model4_fit)
    print(aged_model4_fit)

    np.savetxt('ModelFits/adult_model4_error.txt', np.array([adult_model4_fit.x]))
    np.savetxt('ModelFits/adult_model4_fit.txt', np.array([adult_model4_fit.fun]))
    np.savetxt('ModelFits/aged_model4_error.txt', np.array([aged_model4_fit.x]))
    np.savetxt('ModelFits/aged_model4_fit.txt', np.array([aged_model4_fit.fun]))


    print('Entering model5 fit')
    print('Adult fit')
    adult_model5_fit = sp.optimize.differential_evolution(func=Model5_RMSLE, bounds = Adult_Model5_Parameter_Bounds,args=(19,1000,Adult_Viral_Data,Adult_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
    print(adult_model5_fit)
    print('Aged fit')
    aged_model5_fit = sp.optimize.differential_evolution(func=Model5_RMSLE, bounds = Aged_Model5_Parameter_Bounds,args=(19,1000,Aged_Viral_Data,Aged_CD8_Data),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)

    print(adult_model5_fit)
    print(aged_model5_fit)

    np.savetxt('ModelFits/adult_model5_error.txt', np.array([adult_model5_fit.x]))
    np.savetxt('ModelFits/adult_model5_fit.txt', np.array([adult_model5_fit.fun]))
    np.savetxt('ModelFits/aged_model5_error.txt', np.array([aged_model5_fit.x]))
    np.savetxt('ModelFits/aged_model5_fit.txt', np.array([aged_model5_fit.fun]))

    end = time.time()

    print(f'Time taken: {end-start}')


    print(adult_model1_fit.fun)
    print(aged_model1_fit.fun)
    print(adult_model2_fit.fun)
    print(aged_model2_fit.fun)
    #print(aged_model3_fit.fun)
    #print(adult_model3_fit.fun)
    print(adult_model4_fit.fun)
    print(aged_model4_fit.fun)
    print(adult_model5_fit.fun)
    print(aged_model5_fit.fun)



    adult_model1_params = adult_model1_fit.x
    aged_model1_params = aged_model1_fit.x
    
    adult_model2_params = adult_model2_fit.x
    aged_model2_params = aged_model2_fit.x
    
    #adult_model3_params = adult_model3_fit.x
    #aged_model3_params = aged_model3_fit.x
    
    adult_model4_params = adult_model4_fit.x
    aged_model4_params = aged_model4_fit.x
    
    adult_model5_params = adult_model5_fit.x
    aged_model5_params = aged_model5_fit.x
    
    #adult_model3_predictions = Model3_Predict(adult_model3_params,19,10)
    #aged_model3_predictions = Model3_Predict(aged_model3_params,19,10)
    
    
    adult_model1_AICc = Model1_AICc(adult_model1_params,19,1000,Adult_Viral_Data,Adult_CD8_Data)
    aged_model1_AICc = Model1_AICc(aged_model1_params,19,1000,Aged_Viral_Data,Aged_CD8_Data)
    
    adult_model2_AICc = Model2_AICc(adult_model2_params,19,1000,Adult_Viral_Data,Adult_CD8_Data)
    aged_model2_AICc = Model2_AICc(aged_model2_params,19,1000,Aged_Viral_Data,Aged_CD8_Data)
    
    #adult_model3_AICc = Model3_AICc(adult_model3_params,19,1000,Adult_Viral_Data,Adult_CD8_Data)
    #aged_model3_AICc = Model3_AICc(aged_model3_params,19,1000,Aged_Viral_Data,Aged_CD8_Data)
    
    adult_model4_AICc = Model4_AICc(adult_model4_params,19,1000,Adult_Viral_Data,Adult_CD8_Data)
    aged_model4_AICc = Model4_AICc(aged_model4_params,19,1000,Aged_Viral_Data,Aged_CD8_Data)
    
    adult_model5_AICc = Model5_AICc(adult_model5_params,19,1000,Adult_Viral_Data,Adult_CD8_Data)
    aged_model5_AICc = Model5_AICc(aged_model5_params,19,1000,Aged_Viral_Data,Aged_CD8_Data)
    

    end = time.time()
    
    print(f'Total time: {end-start}')
    
    print(f'Adult Model 1 AICc: {adult_model1_AICc}')
    print(f'Aged Model 1 AICc: {aged_model1_AICc}')
    
    print(f'Adult Model 2 AICc: {adult_model2_AICc}')
    print(f'Aged Model 2 AICc: {aged_model2_AICc}')

    #print(f'Adult Model 3 AICc: {adult_model3_AICc}')
    #print(f'Aged Model 3 AICc: {aged_model3_AICc}')
    
    print(f'Adult Model 4 AICc: {adult_model4_AICc}')
    print(f'Aged Model 4 AICc: {aged_model4_AICc}')
    
    print(f'Adult Model 5 AICc: {adult_model5_AICc}')
    print(f'Aged Model 5 AICc: {aged_model5_AICc}')
    
    
    adult_model1_predictions = Model1_Predict(adult_model1_params, 19,10)
    aged_model1_predictions = Model1_Predict(aged_model1_params, 19,10)
    adult_model2_predictions = Model2_Predict(adult_model2_params, 19,10)
    aged_model2_predictions = Model2_Predict(aged_model2_params, 19,10)
    #adult_model3_predictions = Model3_Predict(adult_model3_params, 19,10)
    #aged_model3_predictions = Model3_Predict(aged_model3_params, 19,10)
    adult_model4_predictions = Model4_Predict(adult_model4_params, 19,10)
    aged_model4_predictions = Model4_Predict(aged_model4_params, 19,10)
    adult_model5_predictions = Model5_Predict(adult_model5_params, 19,10)
    aged_model5_predictions = Model5_Predict(aged_model5_params, 19,10)
    
    
    plt.figure(figsize=(8,6))
    plt.title('Aged Virus w/ Prediction')
    plt.ylabel('Aged Virus per g/tissue')
    plt.xlabel('Days Post Infection')
    plt.yscale('log')
    plt.plot(Aged_Viral_Data['DPI'],Aged_Viral_Data['Viral Titer (Pfu/ml)'],'bo')
    plt.plot(t,aged_model1_predictions[:-1,0],label='Model 1 Predicted')
    plt.plot(t,aged_model2_predictions[:-1,0],label='Model 2 Predicted')
    #plt.plot(t,aged_model3_predictions[:-1,0],label='Model 3 Predicted')
    plt.plot(t,aged_model4_predictions[:-1,0],label='Model 4 Predicted')
    plt.plot(t,aged_model5_predictions[:-1,2],label='Model 5 Predicted')
    #plt.plot(t,aged_model3_cvfixed_predictions[:-1,0],label='Model 3 Predicted, c_V Fixed')
    plt.legend()
    plt.savefig("Aged_Virus_Preds.png")
    #plt.show()
    
    plt.figure(figsize=(8,6))
    plt.title('Adult Virus w/ Prediction')
    plt.ylabel('Adult Virus per g/tissue')
    plt.xlabel('Days Post Infection')
    plt.yscale('log')
    plt.plot(Adult_Viral_Data['DPI'],Adult_Viral_Data['Viral Titer (Pfu/ml)'],'bo')
    plt.plot(t,adult_model1_predictions[:-1,0],label='Model 1 Predicted')
    plt.plot(t,adult_model2_predictions[:-1,0],label='Model 2 Predicted')
    #plt.plot(t,adult_model3_predictions[:-1,0],label='Model 3 Predicted')
    plt.plot(t,adult_model4_predictions[:-1,0],label='Model 4 Predicted')
    plt.plot(t,adult_model5_predictions[:-1,2],label='Model 5 Predicted')
    #plt.plot(t,adult_model3_cvfixed_predictions[:-1,0],label='Model 3 Predicted, c_V Fixed')

    plt.legend()
    plt.savefig("Adult_Virus_Preds.png")
    #plt.show()
    
    #Plot Aged CD8
    plt.figure(figsize=(8,6))
    plt.title('Aged CD8+ w/ Prediction')
    plt.ylabel('Aged CD8+ per g/tissue')
    plt.xlabel('Days Post Infection')
    plt.yscale('log')
    plt.plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],'bo')
#    plt.plot(t[:60],aged_model3_predictions[:60,1],label='Model 3 Predicted')
    plt.plot(t,aged_model1_predictions[:-1,1],label='Model 1 Predicted')
    plt.plot(t,aged_model2_predictions[:-1,1],label='Model 2 Predicted')
    #plt.plot(t,aged_model3_predictions[:-1,2],label='Model 3 Predicted')
    plt.plot(t,aged_model4_predictions[:-1,1],label='Model 4 Predicted')
    plt.plot(t,aged_model5_predictions[:-1,3],label='Model 5 Predicted')

    #plt.plot(t,aged_model3_cvfixed_predictions[:-1,1],label='Model 3 Predicted, c_V Fixed')
    plt.legend()
    plt.savefig("Aged_CD8_Preds.png")
    #plt.show()
    
    plt.figure(figsize=(8,6))
    plt.title('Adult CD8+ w/ Prediction')
    plt.ylabel('Adult CD8+ per g/tissue')
    plt.xlabel('Days Post Infection')
    plt.yscale('log')
    plt.plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],'bo')
    #plt.plot(t[:60],adult_model3_predictions[:60,1], label = 'Model 3 Predicted')
    #plt.plot(t,adult_model3_cvfixed_predictions[:-1,1],label='Model 3 Predicted, c_V Fixed')
    plt.plot(t,adult_model1_predictions[:-1,1],label='Model 1 Predicted')
    plt.plot(t,adult_model2_predictions[:-1,1],label='Model 2 Predicted')
    #plt.plot(t,adult_model3_predictions[:-1,2],label='Model 3 Predicted')
    plt.plot(t,adult_model4_predictions[:-1,1],label='Model 4 Predicted')
    plt.plot(t,adult_model5_predictions[:-1,3],label='Model 5 Predicted')
    plt.legend()
    plt.savefig("Adult_CD8_Preds.png")
    #plt.show()