import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from Models.ModelC1 import *
from Models.ModelC2 import *
from Models.ModelC3 import *
from Models.ModelC4 import *

from ModelBounds import *


plt.set_cmap('tab20')

WORKERS = -1

RESAMPLES=300

POPSIZE=100
MAXITER=500

TOL = 1e-16

#TODO: Some of the following boilerplate can be cleaned up.
CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

Adult_Viral_Data_Resamples = pd.read_csv("BootstrappingData/Adult_Viral_Resamples.csv")
Aged_Viral_Data_Resamples = pd.read_csv("BootstrappingData/Aged_Viral_Resamples.csv")

Adult_CD8_Data_Resamples = pd.read_csv("BootstrappingData/Adult_CD8_Resamples.csv")
Aged_CD8_Data_Resamples = pd.read_csv("BootstrappingData/Aged_CD8_Resamples.csv")


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

V_0 = 25.0

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
    print('starting')
    start = time.time()

    fit_bootstrapping(ModelC1_RMSLE,Adult_ModelC1_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','K','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MC1_Adult_Bootstrapping.csv","Adult_MC1",1)
    #fit_bootstrapping(ModelC1_RMSLE,Aged_ModelC1_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','K','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MC1_Aged_Bootstrapping.csv","Aged_MC1",1)

    #fit_bootstrapping(ModelC2_RMSLE,Adult_ModelC2_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','c_T','K','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MC2_Adult_Bootstrapping.csv","Adult_MC2",3)
    #fit_bootstrapping(ModelC2_RMSLE,Aged_ModelC2_Parameter_Bounds,['p','k_V','c_V','r','k_T','d_T','c_T','K','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MC2_Aged_Bootstrapping.csv","Aged_MC2",3)

    #fit_bootstrapping(ModelC3_RMSLE,Adult_ModelC3_Parameter_Bounds,['p','k_V','c_V','r','d_T','K','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MC3_Adult_Bootstrapping.csv","Adult_MC3",3)
    #fit_bootstrapping(ModelC3_RMSLE,Aged_ModelC3_Parameter_Bounds,['p','k_V','c_V','r','d_T','K','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MC3_Aged_Bootstrapping.csv","Aged_MC3",3)

    #fit_bootstrapping(ModelC4_RMSLE,Adult_ModelC4_Parameter_Bounds,['p','k_V','c_V','r','d_T','c_T','K','T_0','V_0'],Adult_Viral_Data_Resamples,Adult_CD8_Data_Resamples,"BootstrappingFits/MC4_Adult_Bootstrapping.csv","Adult_MC4",3)
    #fit_bootstrapping(ModelC4_RMSLE,Aged_ModelC4_Parameter_Bounds,['p','k_V','c_V','r','d_T','c_T','K','T_0','V_0'],Aged_Viral_Data_Resamples,Aged_CD8_Data_Resamples,"BootstrappingFits/MC4_Aged_Bootstrapping.csv","Aged_MC4",3)

    end = time.time()
    print(f'Time Taken: {end-start}')
