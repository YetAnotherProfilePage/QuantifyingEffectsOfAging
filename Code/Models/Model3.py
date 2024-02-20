#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 23:02:04 2023

@author: benjaminwhipple
"""

import numpy as np
import scipy as sp

'''
def model5(y,t,p,k_V,c_V,gamma,k_N,d_N,s_T,r,k_T,c_T,d_T,L_T):
    V,N,T = y
    
    dVdt = p*V*(1-V/k_V)-c_V*V*T
    dNdt = gamma*(V/V+k_N) - d_N*N
    dTdt = s_T + r*T*(N/(N+k_T))-c_T*((1/(1 + N**2)))*T - d_T*T
    
    dydt = [dVdt,dNdt,dTdt]
    return dydt
'''

def model3(y,t,p,k_V,c_V,s_N,gamma,d_N,s_T,r,k_T,c_T,d_T):
    V,N,T = y
    
    dVdt = p*V*(1-V/k_V)-c_V*V*T
    dNdt = s_N + gamma*V*N - d_N*N
    dTdt = s_T + r*T*(N/(N+k_T))-d_T*((1/(1 + N**2)))*T - c_T*T
    #dVdt = p*y[0]*(1-y[0]/k_V)-c_V*y[0]*y[2]
    #dNdt = s_N + gamma*y[0]*y[1] - d_N*y[1]
    #dTdt = s_T + r*y[2]*(y[1]/(y[1]+k_T))-d_T*((1/(1 + y[1]**2)))*y[0] - c_T*y[2]
    dydt = [dVdt,dNdt,dTdt]
    return dydt

#Should probably rename to Model5_Loss
def Model3_RMSLE(x,times,step_res,Viral_Data,CD8_Data,
                 Viral_Data_Col='Viral Titer (Pfu/ml)',
                 CD8_Data_Col='CD8+ per g/tissue',time_col='DPI'
                ):
    #This function hard codes column names. DPI represents days past infection column. Fix in future.

    #p,k_V,c_V,s_N,gamma,d_N,r,k_T,c_T,d_T,L_T,T_0,V_0 = x #This is the order the parameters should be input.
    N_0 = 1.0
    p,k_V,c_V,gamma,d_N,r,k_T,c_T,d_T,T_0,V_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    #s_N = d_N*N_0
    s_N = d_N*N_0
    s_T = c_T  * T_0 + d_T * T_0
    y0 = [V_0, N_0 ,T_0]

    t = np.linspace(0,times,times*step_res+1)
    #p,k_V,c_V,s_T,r,k_T,c_T

    predictions = sp.integrate.odeint(model3,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_N,gamma,d_N,s_T,r,k_T,c_T,d_T)
                                        )[0:step_res*times+1:step_res,:]
    #print('here')
    """
    predictions = sp.integrate.solve_ivp(model3,
                                         [t[0],t[-1]],
                                         y0,
                                         method='RK4',
                                         args=(p,k_V,c_V,s_N,gamma,d_N,s_T,r,k_T,c_T,d_T)
                                        )[0:step_res*times+1:step_res,:]
    """
    num_observations = len(Viral_Data) + len(CD8_Data)

    #Need to compute SSE across all observations, for each day.

    Viral_Pred_SSE = 0.0
    
    #'''

    for i in Viral_Data[time_col].unique():
        temp = Viral_Data[Viral_Data[time_col] == i][Viral_Data_Col].to_numpy()

        Viral_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,0]+1)))
    
    #'''
    CD8_Pred_SSE = 0.0

    for i in CD8_Data[time_col].unique():
        temp = CD8_Data[CD8_Data[time_col] == i][CD8_Data_Col].to_numpy()
        CD8_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,2]+1)))
    
    RMSE = np.sqrt((1/num_observations)*(2*CD8_Pred_SSE + Viral_Pred_SSE))

    return RMSE

def Model3_AICc(x,times,step_res,Viral_Data,CD8_Data,
                 Viral_Data_Col='Viral Titer (Pfu/ml)',
                 CD8_Data_Col='CD8+ per g/tissue',time_col='DPI'
                ):
    #This function hard codes column names. DPI represents days past infection column. Fix in future.

    #p,k_V,c_V,s_N,gamma,d_N,r,k_T,c_T,d_T,L_T,T_0,V_0 = x #This is the order the parameters should be input.
    p,k_V,c_V,gamma,d_N,r,k_T,c_T,d_T,T_0,V_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    N_0 = 1
    s_N = d_N*N_0
    s_T = c_T  * T_0 + d_T * T_0
    y0 = [V_0, N_0,T_0]

    t = np.linspace(0,times,times*step_res+1)
    #p,k_V,c_V,s_T,r,k_T,c_T

    predictions = sp.integrate.odeint(model3,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_N,gamma,d_N,s_T,r,k_T,c_T,d_T)
                                        )[0:step_res*times+1:step_res,:]

    num_observations = len(Viral_Data) + len(CD8_Data)
    num_params = 6

    #Need to compute SSE across all observations, for each day.

    Viral_Pred_SSE = 0.0
    
    #'''

    for i in Viral_Data[time_col].unique():
        temp = Viral_Data[Viral_Data[time_col] == i][Viral_Data_Col].to_numpy()

        Viral_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,0]+1)))
    
    #'''
    CD8_Pred_SSE = 0.0

    for i in CD8_Data[time_col].unique():
        temp = CD8_Data[CD8_Data[time_col] == i][CD8_Data_Col].to_numpy()
        CD8_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,2]+1)))
    
    RSS = (CD8_Pred_SSE + Viral_Pred_SSE)
    AICc = num_observations*np.log(RSS/num_observations)+(2*num_observations*num_params)/(num_observations - num_params - 1)

    return AICc

def Model3_Predict(x,times,step_res):
    p,k_V,c_V,gamma,d_N,r,k_T,c_T,d_T,T_0,V_0 = x 
    s_T = c_T  * T_0 + d_T * T_0
    N_0 = 1
    s_N = d_N*N_0
    
    y0 = [V_0, N_0, T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(model3,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_N,gamma,d_N,s_T,r,k_T,c_T,d_T),
                                         printmessg=False
                                        )
    #[0:step_res*times+1:step_res,:]
    return predictions
