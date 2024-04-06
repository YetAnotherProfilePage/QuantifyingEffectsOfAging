import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from Models.ModelA1 import *
from Models.ModelA2 import *
from Models.ModelA3 import *
from Models.ModelA4 import *


plt.set_cmap('tab20')

WORKERS = -1

RESAMPLES=300

POPSIZE=100
MAXITER=500

#POPSIZE=5
#MAXITER=5


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


Adult_Viral_Data_Resamples = pd.read_csv("BootstrappingData/Adult_Viral_Resamples.csv")
Aged_Viral_Data_Resamples = pd.read_csv("BootstrappingData/Aged_Viral_Resamples.csv")

Adult_CD8_Data_Resamples = pd.read_csv("BootstrappingData/Adult_CD8_Resamples.csv")
Aged_CD8_Data_Resamples = pd.read_csv("BootstrappingData/Aged_CD8_Resamples.csv")

T_0_Adult = np.mean(Adult_CD8_Data[Adult_CD8_Data['DPI']==0]['CD8+ per g/tissue'])
T_0_Aged = np.mean(Aged_CD8_Data[Aged_CD8_Data['DPI']==0]['CD8+ per g/tissue'])

print(T_0_Adult)

t = np.linspace(0,19,190)

#Note: We do not distinguish k_V between Aged and Adult.
k_V = np.max(Viral_Data['Viral Titer (Pfu/ml)']) #k_V = 1.2e6

#print(np.max(CD8_Data['CD8+ per g/tissue'])) #Order 1e7

V_0 = 25.0

Adult_ModelA1_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA1_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Adult_ModelA2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA2_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4,1e2),
    (1e1,1e8),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelA3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA3_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]

Adult_ModelA4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1.0/k_V,r*1.0/k_V),
    #(r*1e-10,r*1e-10),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Adult,T_0_Adult),
    (V_0,V_0), #V_0
    ]

Aged_ModelA4_Parameter_Bounds = [
    (1e-2,1e2), #p
    (k_V,k_V), #k_V #1.2e6
    (1e-8,1e-4), #c_V
    #(r,r), #r
    (1e-4/k_V,1e2/k_V),
    #(r*1e-12,r*1e-12),
    (1e-4,1), #d_T
    (1e-4,1), #c_T
    (T_0_Aged,T_0_Aged),
    (V_0,V_0), #V_0
    ]


def trim_to_threshhold(data, threshold = 25.0, minimum=True):
    #Utility to censor data at threshold.
    if minimum==True:
        return np.where(data<threshold,threshold,data)
    else:
        return np.where(data>threshold,threshold,data)

def fit_bootstrapping(model_loss, model_bounds, parameters, VirusData_Resamples, CD8Data_Resamples, file_name, model_name,repeats = 1,time_col="DPI"):
    
    results = []

    cols = ['Resample','RMSLE']+parameters
    print(cols)

    total_start = time.time()
    for i in range(RESAMPLES):
        current_resample = "Resample"+str(i)
        #print(VirusData_Resamples)
        #print(CD8Data_Resamples)

        #print(VirusData_Resamples.columns)

        #print(VirusData_Resamples[current_resample])
        print(f'Resample {i}')
        #print(current_resample)
        these_fits = []
        scores = []
        for j in range(repeats):
            print(f'In {model_name}, In resample {i}, In repeat {j+1}/{repeats}')
            
            #Get index of T_0
            T_0_indx = parameters.index("T_0")

            resampled_T_0 = np.mean(CD8Data_Resamples[CD8Data_Resamples[time_col]==0][f"Resample{i}"])
            print(resampled_T_0)
            model_bounds[T_0_indx]=(resampled_T_0,resampled_T_0)

            this_fit = sp.optimize.differential_evolution(func=model_loss, bounds = model_bounds,args=(19,1000,VirusData_Resamples,CD8Data_Resamples,current_resample,current_resample,'DPI'),popsize=POPSIZE,updating='deferred',workers=WORKERS,maxiter=MAXITER,tol = TOL,polish=False)
            these_fits.append(this_fit)
            scores.append(this_fit.fun)

        fit = list(these_fits[np.argmin(scores)].x)
        loss = np.min(scores)

        results.append([i+1,loss]+fit)

    #print(results)
    temp = pd.DataFrame(data=results,columns=cols,dtype=np.float64)
    print(temp)
    temp.to_csv(file_name)

if __name__ == "__main__":
    start = time.time()
    fit_bootstrapping(ModelA1_RMSLE,Adult_ModelA1_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MA1_Adult_Bootstrapping.csv","Adult_MA1",3)
    fit_bootstrapping(ModelA1_RMSLE,Aged_ModelA1_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MA1_Aged_Bootstrapping.csv","Aged_MA1",3)

    fit_bootstrapping(ModelA2_RMSLE,Adult_ModelA2_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','c_T','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MA2_Adult_Bootstrapping.csv","Adult_MA2",3)
    fit_bootstrapping(ModelA2_RMSLE,Aged_ModelA2_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','c_T','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MA2_Aged_Bootstrapping.csv","Aged_MA2",3)

    fit_bootstrapping(ModelA3_RMSLE,Adult_ModelA3_Parameter_Bounds,['p','k_V','c_V','r','d_T','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MA3_Adult_Bootstrapping.csv","Adult_MA3",3)
    fit_bootstrapping(ModelA3_RMSLE,Aged_ModelA3_Parameter_Bounds,['p','k_V','c_V','r','d_T','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MA3_Aged_Bootstrapping.csv","Aged_MA3",3)

    fit_bootstrapping(ModelA4_RMSLE,Adult_ModelA4_Parameter_Bounds,['p','k_V','c_V','r','d_T','c_T','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MA4_Adult_Bootstrapping.csv","Adult_MA4",3)
    fit_bootstrapping(ModelA4_RMSLE,Aged_ModelA4_Parameter_Bounds,['p','k_V','c_V','r','d_T','c_T','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MA4_Aged_Bootstrapping.csv","Aged_MA4",3)

    end = time.time()

    print(f'Time taken: {end-start}')
