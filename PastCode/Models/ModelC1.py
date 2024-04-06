#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 23:26:34 2023

@author: benjaminwhipple

Models C_ are Models A_ with additional regulation of Viral replication by T cell presence by multiplicative (K^2/(K^2 + T^2))
"""

import numpy as np
import scipy as sp

def modelC1(y,t,p,k_V,c_V,s_T,r,k_T,c_T,K):
    V,T = y

    dVdt = p*V*(1-V/k_V)*(1/(1+(K*T)**2))-c_V*V*T
    dTdt = s_T + r*T*(V/(V+k_T))-c_T*T
    
    if V < 0:
        dVdt = 0

    dydt = [dVdt,dTdt]
    return dydt

def ModelC1_RMSLE(x,times,step_res,Viral_Data,CD8_Data,
                 Viral_Data_Col='Viral Titer (Pfu/ml)',
                 CD8_Data_Col='CD8+ per g/tissue',time_col='DPI'
                ):
    #This function hard codes column names. DPI represents days past infection column. Fix in future.

    p,k_V,c_V,r,k_T,c_T,K,T_0,V_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    
    s_T = c_T  * T_0
    y0 = [V_0, T_0]

    t = np.linspace(0,times,times*step_res+1)
    #p,k_V,c_V,s_T,r,k_T,c_T,K

    predictions = sp.integrate.odeint(modelC1,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_T,r,k_T,c_T,K)
                                        )[0:step_res*times+1:step_res,:]

    num_observations = len(Viral_Data) + len(CD8_Data)

    #Need to compute SSE across all observations, for each day.

    Viral_Pred_SSE = 0.0

    for i in Viral_Data[time_col].unique():
        temp = Viral_Data[Viral_Data[time_col] == i][Viral_Data_Col].to_numpy()

        Viral_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,0]+1)))
        
    CD8_Pred_SSE = 0.0

    for i in CD8_Data[time_col].unique():
        temp = CD8_Data[CD8_Data[time_col] == i][CD8_Data_Col].to_numpy()
        CD8_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,1]+1)))
    
    RMSE = np.sqrt((1/num_observations)*(5*CD8_Pred_SSE + Viral_Pred_SSE))

    return RMSE

def ModelC1_AICc(x,times,step_res,Viral_Data,CD8_Data,
                 Viral_Data_Col='Viral Titer (Pfu/ml)',
                 CD8_Data_Col='CD8+ per g/tissue',time_col='DPI'
                ):
    p,k_V,c_V,r,k_T,c_T,K,T_0,V_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.

    # Params fit: p, c_V, r, k_T, c_T = 6
    
    s_T = c_T  * T_0
    y0 = [V_0, T_0]

    t = np.linspace(0,times,times*step_res+1)
    #p,k_V,c_V,s_T,r,k_T,c_T

    predictions = sp.integrate.odeint(modelC1,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_T,r,k_T,c_T,K)
                                        )[0:step_res*times+1:step_res,:]

    num_observations = len(Viral_Data) + len(CD8_Data)
    #num_params = 3
    num_params = 6

    Viral_Pred_SSE = 0.0

    for i in Viral_Data[time_col].unique():
        temp = Viral_Data[Viral_Data[time_col] == i][Viral_Data_Col].to_numpy()

        Viral_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,0]+1)))
        
    CD8_Pred_SSE = 0.0

    for i in CD8_Data[time_col].unique():
        temp = CD8_Data[CD8_Data[time_col] == i][CD8_Data_Col].to_numpy()
        CD8_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,1]+1)))
    
    RSS = (CD8_Pred_SSE + Viral_Pred_SSE)
    
    AICc = num_observations*np.log(RSS/num_observations)+(2*num_observations*num_params)/(num_observations - num_params - 1)

    return AICc

def ModelC1_Predict(x,times,step_res):
    p,k_V,c_V,r,k_T,c_T,K,T_0,V_0 = x
    s_T = c_T*T_0
    y0 = [V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelC1,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_T,r,k_T,c_T,K),
                                         printmessg=False
                                        )
    #[0:step_res*times+1:step_res,:]
    return predictions

def ModelC1_ViralClearanceTime(x,times,step_res,threshhold = 10.0):
    p,k_V,c_V,r,k_T,c_T,K,T_0,V_0 = x
    s_T = c_T*T_0
    y0 = [V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelC1,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_T,r,k_T,c_T,K),
                                         printmessg=False
                                        )
    valid_indices = np.where(predictions[:,0]>threshhold)
    virus_times = t[valid_indices]
    return virus_times[-1]

def ModelC1_TotalViralLoad(x,times,step_res,threshhold = 10.0):
    p,k_V,c_V,r,k_T,c_T,K,T_0,V_0 = x
    s_T = c_T*T_0
    y0 = [V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelC1,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_T,r,k_T,c_T,K),
                                         printmessg=False
                                        )
    valid_indices = np.where(predictions[:,0]>threshhold)
    virus_times = t[valid_indices]
    
    dt = t[2]-t[1]
    total = np.sum(predictions[valid_indices,0][0]*dt)
    return total

def ModelC1_ExcessCTL(x,times,step_res,threshhold = 10.0):
    p,k_V,c_V,r,k_T,c_T,K,T_0,V_0 = x
    s_T = c_T*T_0
    y0 = [V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelC1,
                                         y0,
                                         t,
                                         args=(p,k_V,c_V,s_T,r,k_T,c_T,K),
                                         printmessg=False
                                        )
    valid_indices = np.where(predictions[:,0]>threshhold)
    virus_times = t[valid_indices]
    
    dt = t[2]-t[1]

    total = sum((predictions[valid_indices[0][-1]:,1]-T_0) * dt)
    return total