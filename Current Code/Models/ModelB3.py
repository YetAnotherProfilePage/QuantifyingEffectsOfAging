#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 23:00:58 2023

@author: benjaminwhipple
"""

import numpy as np
import scipy as sp

def modelB3(y,t,beta,d_I,p,c,s_T,d_T,r,k_T):
    U,I,V,T = y
    
    dUdt = -beta*U*V
    dIdt = beta*U*V-d_I*T*I
    dVdt = p*I-c*V
    dTdt = s_T - d_T*T + r*T*(V/(V+k_T))
    
    dydt = [dUdt,dIdt,dVdt,dTdt]
    return dydt

def ModelB3_RMSLE(x,times,step_res,Viral_Data,CD8_Data,
                 Viral_Data_Col='Viral Titer (Pfu/ml)',
                 CD8_Data_Col='CD8+ per g/tissue',time_col='DPI'
                ):
    #This function hard codes column names. DPI represents days past infection column. Fix in future.

    beta,d_I,p,c,d_T,r,k_T,U_0,I_0,V_0,T_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    
    s_T = d_T*T_0
    y0 = [U_0,I_0,V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)
    #p,k_V,c_V,s_T,r,k_T,c_T

    predictions = sp.integrate.odeint(modelB3,
                                         y0,
                                         t,
                                         args=(beta,d_I,p,c,s_T,d_T,r,k_T)
                                        )[0:step_res*times+1:step_res,:]

    num_observations = len(Viral_Data) + len(CD8_Data)

    #Need to compute SSE across all observations, for each day.

    Viral_Pred_SSE = 0.0

    for i in Viral_Data[time_col].unique():
        temp = Viral_Data[Viral_Data[time_col] == i][Viral_Data_Col].to_numpy()

        Viral_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,2]+1)))
        
    CD8_Pred_SSE = 0.0

    for i in CD8_Data[time_col].unique():
        temp = CD8_Data[CD8_Data[time_col] == i][CD8_Data_Col].to_numpy()
        CD8_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,3]+1)))
    
    RMSE = np.sqrt((1/num_observations)*(5*CD8_Pred_SSE + Viral_Pred_SSE))

    return RMSE

def ModelB3_AICc(x,times,step_res,Viral_Data,CD8_Data,
                 Viral_Data_Col='Viral Titer (Pfu/ml)',
                 CD8_Data_Col='CD8+ per g/tissue',time_col='DPI',param_modifier=0
                ):
    #This function hard codes column names. DPI represents days past infection column. Fix in future.

    beta,d_I,p,c,d_T,r,k_T,U_0,I_0,V_0,T_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    
    s_T = d_T*T_0
    y0 = [U_0,I_0,V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)
    #p,k_V,c_V,s_T,r,k_T,c_T

    predictions = sp.integrate.odeint(modelB3,
                                         y0,
                                         t,
                                         args=(beta,d_I,p,c,s_T,d_T,r,k_T)
                                        )[0:step_res*times+1:step_res,:]

    num_observations = len(Viral_Data) + len(CD8_Data)
    num_params = 6 - param_modifier
    
    #Need to compute SSE across all observations, for each day.

    Viral_Pred_SSE = 0.0

    for i in Viral_Data[time_col].unique():
        temp = Viral_Data[Viral_Data[time_col] == i][Viral_Data_Col].to_numpy()

        Viral_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,2]+1)))
        
    CD8_Pred_SSE = 0.0

    for i in CD8_Data[time_col].unique():
        temp = CD8_Data[CD8_Data[time_col] == i][CD8_Data_Col].to_numpy()
        CD8_Pred_SSE += np.sum(np.square(np.log(temp+1) - np.log(predictions[i,3]+1)))
    
    RSS = (CD8_Pred_SSE + Viral_Pred_SSE)

    AICc = num_observations*np.log(RSS/num_observations)+(2*num_observations*num_params)/(num_observations - num_params - 1)

    return AICc


def ModelB3_Predict(x,times,step_res):
    beta,d_I,p,c,d_T,r,k_T,U_0,I_0,V_0,T_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    
    s_T = d_T*T_0
    y0 = [U_0,I_0,V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelB3,
                                         y0,
                                         t,
                                         args=(beta,d_I,p,c,s_T,d_T,r,k_T),
                                         printmessg=False
                                        )
    #[0:step_res*times+1:step_res,:]
    return predictions

def ModelB3_ViralClearanceTime(x,times,step_res,threshhold = 10.0):
    beta,d_I,p,c,d_T,r,k_T,U_0,I_0,V_0,T_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    
    s_T = d_T*T_0
    y0 = [U_0,I_0,V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelB3,
                                         y0,
                                         t,
                                         args=(beta,d_I,p,c,s_T,d_T,r,k_T),
                                         printmessg=False
                                        )

    valid_indices = np.where(predictions[:,2]>threshhold)
    virus_times = t[valid_indices]
    return virus_times[-1]

def ModelB3_TotalViralLoad(x,times,step_res,threshhold = 10.0):
    beta,d_I,p,c,d_T,r,k_T,U_0,I_0,V_0,T_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    
    s_T = d_T*T_0
    y0 = [U_0,I_0,V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelB3,
                                         y0,
                                         t,
                                         args=(beta,d_I,p,c,s_T,d_T,r,k_T),
                                         printmessg=False
                                        )

    valid_indices = np.where(predictions[:,2]>threshhold)
    virus_times = t[valid_indices]
    
    dt = t[2]-t[1]
    total = np.sum(predictions[valid_indices,2][0]*dt)
    return total

def ModelB3_ExcessCTL(x,times,step_res,threshhold = 10.0):
    beta,d_I,p,c,d_T,r,k_T,U_0,I_0,V_0,T_0 = x #This is the order the parameters should be input.
    #We leave all parameters as inputs in order to do sensitivity analysis later.
    
    s_T = d_T*T_0
    y0 = [U_0,I_0,V_0,T_0]

    t = np.linspace(0,times,times*step_res+1)

    predictions = sp.integrate.odeint(modelB3,
                                         y0,
                                         t,
                                         args=(beta,d_I,p,c,s_T,d_T,r,k_T),
                                         printmessg=False
                                        )

    valid_indices = np.where(predictions[:,2]>threshhold)
    virus_times = t[valid_indices]
    
    dt = t[2]-t[1]
    
    total = sum((predictions[valid_indices[0][-1]:,3]-T_0) * dt)
    return total
