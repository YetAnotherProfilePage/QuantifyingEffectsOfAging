import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from GroupColors import *

from Models.ModelA1 import *
from Models.ModelA2 import *
from Models.ModelA3 import *
from Models.ModelA4 import *

from Models.ModelB1 import *
from Models.ModelB2 import *
from Models.ModelB3 import *
from Models.ModelB4 import *

from Models.ModelC1 import *
from Models.ModelC2 import *
from Models.ModelC3 import *
from Models.ModelC4 import *

from Models.ModelD1 import *
from Models.ModelD2 import *
from Models.ModelD3 import *
from Models.ModelD4 import *


CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0

Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']

ALPHA = 0.5
t = np.linspace(0,19,190)
t_vals = [i for i in range(1,20)]


VIRAL_THRESHOLD = 24.0

### Load model fits

## Group A

adult_modelA1_params = np.loadtxt("ModelFits/adult_modelA1_params.txt")
aged_modelA1_params = np.loadtxt("ModelFits/aged_modelA1_params.txt")

adult_modelA2_params = np.loadtxt("ModelFits/adult_modelA2_params.txt")
aged_modelA2_params = np.loadtxt("ModelFits/aged_modelA2_params.txt")

adult_modelA3_params = np.loadtxt("ModelFits/adult_modelA3_params.txt")
aged_modelA3_params = np.loadtxt("ModelFits/aged_modelA3_params.txt")

adult_modelA4_params = np.loadtxt("ModelFits/adult_modelA4_params.txt")
aged_modelA4_params = np.loadtxt("ModelFits/aged_modelA4_params.txt")

adult_modelA1_predictions = ModelA1_Predict(adult_modelA1_params, 19,10)
aged_modelA1_predictions = ModelA1_Predict(aged_modelA1_params, 19,10)

adult_modelA2_predictions = ModelA2_Predict(adult_modelA2_params, 19,10)
aged_modelA2_predictions = ModelA2_Predict(aged_modelA2_params, 19,10)

adult_modelA3_predictions = ModelA3_Predict(adult_modelA3_params, 19,10)
aged_modelA3_predictions = ModelA3_Predict(aged_modelA3_params, 19,10)

adult_modelA4_predictions = ModelA4_Predict(adult_modelA4_params, 19,10)
aged_modelA4_predictions = ModelA4_Predict(aged_modelA4_params, 19,10)


AdultA1_valid_indices = np.where(adult_modelA1_predictions[:,0]>VIRAL_THRESHOLD)
AdultA1_Virus_times = t[AdultA1_valid_indices]

AgedA1_valid_indices = np.where(aged_modelA1_predictions[:,0]>VIRAL_THRESHOLD)
AgedA1_Virus_times = t[AgedA1_valid_indices]


AdultA2_valid_indices = np.where(adult_modelA2_predictions[:,0]>VIRAL_THRESHOLD)
AdultA2_Virus_times = t[AdultA2_valid_indices]

AgedA2_valid_indices = np.where(aged_modelA2_predictions[:,0]>VIRAL_THRESHOLD)
AgedA2_Virus_times = t[AgedA2_valid_indices]


AdultA3_valid_indices = np.where(adult_modelA3_predictions[:,0]>VIRAL_THRESHOLD)
AdultA3_Virus_times = t[AdultA3_valid_indices]

AgedA3_valid_indices = np.where(aged_modelA3_predictions[:,0]>VIRAL_THRESHOLD)
AgedA3_Virus_times = t[AgedA3_valid_indices]


AdultA4_valid_indices = np.where(adult_modelA4_predictions[:,0]>VIRAL_THRESHOLD)
AdultA4_Virus_times = t[AdultA4_valid_indices]

AgedA4_valid_indices = np.where(aged_modelA4_predictions[:,0]>VIRAL_THRESHOLD)
AgedA4_Virus_times = t[AgedA4_valid_indices]

## GROUP B

adult_modelB1_params = np.loadtxt("ModelFits/adult_modelB1_params.txt")
aged_modelB1_params = np.loadtxt("ModelFits/aged_modelB1_params.txt")

adult_modelB2_params = np.loadtxt("ModelFits/adult_modelB2_params.txt")
aged_modelB2_params = np.loadtxt("ModelFits/aged_modelB2_params.txt")

adult_modelB3_params = np.loadtxt("ModelFits/adult_modelB3_params.txt")
aged_modelB3_params = np.loadtxt("ModelFits/aged_modelB3_params.txt")

adult_modelB4_params = np.loadtxt("ModelFits/adult_modelB4_params.txt")
aged_modelB4_params = np.loadtxt("ModelFits/aged_modelB4_params.txt")

print(aged_modelB4_params)
print(adult_modelB4_params)

adult_modelB1_predictions = ModelB1_Predict(adult_modelB1_params, 19,10)
aged_modelB1_predictions = ModelB1_Predict(aged_modelB1_params, 19,10)

adult_modelB2_predictions = ModelB2_Predict(adult_modelB2_params, 19,10)
aged_modelB2_predictions = ModelB2_Predict(aged_modelB2_params, 19,10)

adult_modelB3_predictions = ModelB3_Predict(adult_modelB3_params, 19,10)
aged_modelB3_predictions = ModelB3_Predict(aged_modelB3_params, 19,10)

adult_modelB4_predictions = ModelB4_Predict(adult_modelB4_params, 19,10)
aged_modelB4_predictions = ModelB4_Predict(aged_modelB4_params, 19,10)


AdultB1_valid_indices = np.where(adult_modelB1_predictions[:,2]>VIRAL_THRESHOLD)
AdultB1_Virus_times = t[AdultB1_valid_indices]

AgedB1_valid_indices = np.where(aged_modelB1_predictions[:,2]>VIRAL_THRESHOLD)
AgedB1_Virus_times = t[AgedB1_valid_indices]


AdultB2_valid_indices = np.where(adult_modelB2_predictions[:,2]>VIRAL_THRESHOLD)
AdultB2_Virus_times = t[AdultB2_valid_indices]

AgedB2_valid_indices = np.where(aged_modelB2_predictions[:,2]>VIRAL_THRESHOLD)
AgedB2_Virus_times = t[AgedB2_valid_indices]


AdultB3_valid_indices = np.where(adult_modelB3_predictions[:,2]>VIRAL_THRESHOLD)
AdultB3_Virus_times = t[AdultB3_valid_indices]

AgedB3_valid_indices = np.where(aged_modelB3_predictions[:,2]>VIRAL_THRESHOLD)
AgedB3_Virus_times = t[AgedB3_valid_indices]


AdultB4_valid_indices = np.where(adult_modelB4_predictions[:,2]>VIRAL_THRESHOLD)
AdultB4_Virus_times = t[AdultB4_valid_indices]

AgedB4_valid_indices = np.where(aged_modelB4_predictions[:,2]>VIRAL_THRESHOLD)
AgedB4_Virus_times = t[AgedB4_valid_indices]

## GROUP C

adult_modelC1_params = np.loadtxt("ModelFits/adult_modelC1_params.txt")
aged_modelC1_params = np.loadtxt("ModelFits/aged_modelC1_params.txt")

adult_modelC2_params = np.loadtxt("ModelFits/adult_modelC2_params.txt")
aged_modelC2_params = np.loadtxt("ModelFits/aged_modelC2_params.txt")

adult_modelC3_params = np.loadtxt("ModelFits/adult_modelC3_params.txt")
aged_modelC3_params = np.loadtxt("ModelFits/aged_modelC3_params.txt")

adult_modelC4_params = np.loadtxt("ModelFits/adult_modelC4_params.txt")
aged_modelC4_params = np.loadtxt("ModelFits/aged_modelC4_params.txt")

adult_modelC1_predictions = ModelC1_Predict(adult_modelC1_params, 19,10)
aged_modelC1_predictions = ModelC1_Predict(aged_modelC1_params, 19,10)

adult_modelC2_predictions = ModelC2_Predict(adult_modelC2_params, 19,10)
aged_modelC2_predictions = ModelC2_Predict(aged_modelC2_params, 19,10)

adult_modelC3_predictions = ModelC3_Predict(adult_modelC3_params, 19,10)
aged_modelC3_predictions = ModelC3_Predict(aged_modelC3_params, 19,10)

adult_modelC4_predictions = ModelC4_Predict(adult_modelC4_params, 19,10)
aged_modelC4_predictions = ModelC4_Predict(aged_modelC4_params, 19,10)



AdultC1_valid_indices = np.where(adult_modelC1_predictions[:,0]>VIRAL_THRESHOLD)
AdultC1_Virus_times = t[AdultC1_valid_indices]

AgedC1_valid_indices = np.where(aged_modelC1_predictions[:,0]>VIRAL_THRESHOLD)
AgedC1_Virus_times = t[AgedC1_valid_indices]


AdultC2_valid_indices = np.where(adult_modelC2_predictions[:,0]>VIRAL_THRESHOLD)
AdultC2_Virus_times = t[AdultC2_valid_indices]

AgedC2_valid_indices = np.where(aged_modelC2_predictions[:,0]>VIRAL_THRESHOLD)
AgedC2_Virus_times = t[AgedC2_valid_indices]


AdultC3_valid_indices = np.where(adult_modelC3_predictions[:,0]>VIRAL_THRESHOLD)
AdultC3_Virus_times = t[AdultC3_valid_indices]

AgedC3_valid_indices = np.where(aged_modelC3_predictions[:,0]>VIRAL_THRESHOLD)
AgedC3_Virus_times = t[AgedC3_valid_indices]


AdultC4_valid_indices = np.where(adult_modelC4_predictions[:,0]>VIRAL_THRESHOLD)
AdultC4_Virus_times = t[AdultC4_valid_indices]

AgedC4_valid_indices = np.where(aged_modelC4_predictions[:,0]>VIRAL_THRESHOLD)
AgedC4_Virus_times = t[AgedC4_valid_indices]

## GROUP D

adult_modelD1_params = np.loadtxt("ModelFits/adult_modelD1_params.txt")
aged_modelD1_params = np.loadtxt("ModelFits/aged_modelD1_params.txt")

adult_modelD2_params = np.loadtxt("ModelFits/adult_modelD2_params.txt")
aged_modelD2_params = np.loadtxt("ModelFits/aged_modelD2_params.txt")

adult_modelD3_params = np.loadtxt("ModelFits/adult_modelD3_params.txt")
aged_modelD3_params = np.loadtxt("ModelFits/aged_modelD3_params.txt")

adult_modelD4_params = np.loadtxt("ModelFits/adult_modelD4_params.txt")
aged_modelD4_params = np.loadtxt("ModelFits/aged_modelD4_params.txt")

adult_modelD1_predictions = ModelD1_Predict(adult_modelD1_params, 19,10)
aged_modelD1_predictions = ModelD1_Predict(aged_modelD1_params, 19,10)

adult_modelD2_predictions = ModelD2_Predict(adult_modelD2_params, 19,10)
aged_modelD2_predictions = ModelD2_Predict(aged_modelD2_params, 19,10)

adult_modelD3_predictions = ModelD3_Predict(adult_modelD3_params, 19,10)
aged_modelD3_predictions = ModelD3_Predict(aged_modelD3_params, 19,10)

adult_modelD4_predictions = ModelD4_Predict(adult_modelD4_params, 19,10)
aged_modelD4_predictions = ModelD4_Predict(aged_modelD4_params, 19,10)


AdultD1_valid_indices = np.where(adult_modelD1_predictions[:,2]>VIRAL_THRESHOLD)
AdultD1_Virus_times = t[AdultD1_valid_indices]

AgedD1_valid_indices = np.where(aged_modelD1_predictions[:,2]>VIRAL_THRESHOLD)
AgedD1_Virus_times = t[AgedD1_valid_indices]


AdultD2_valid_indices = np.where(adult_modelD2_predictions[:,2]>VIRAL_THRESHOLD)
AdultD2_Virus_times = t[AdultD2_valid_indices]

AgedD2_valid_indices = np.where(aged_modelD2_predictions[:,2]>VIRAL_THRESHOLD)
AgedD2_Virus_times = t[AgedD2_valid_indices]


AdultD3_valid_indices = np.where(adult_modelD3_predictions[:,2]>VIRAL_THRESHOLD)
AdultD3_Virus_times = t[AdultD3_valid_indices]

AgedD3_valid_indices = np.where(aged_modelD3_predictions[:,2]>VIRAL_THRESHOLD)
AgedD3_Virus_times = t[AgedD3_valid_indices]


AdultD4_valid_indices = np.where(adult_modelD4_predictions[:,2]>VIRAL_THRESHOLD)
AdultD4_Virus_times = t[AdultD4_valid_indices]

AgedD4_valid_indices = np.where(aged_modelD4_predictions[:,2]>VIRAL_THRESHOLD)
AgedD4_Virus_times = t[AgedD4_valid_indices]

# Clip data:
# Adult data is only valid through day 9

Plottable_Adult_Viral_Data = Adult_Viral_Data[Adult_Viral_Data["DPI"]<=9]

# Aged data is only valid through day 11
Plottable_Aged_Viral_Data = Aged_Viral_Data[Aged_Viral_Data["DPI"]<=11]


if __name__ == "__main__":
        
    ## GROUP A PLOTTING.

    plt.figure()

    fig, axs = plt.subplots(2, 2,figsize=(13,6),sharex='col',sharey='row')

    #axs[0, 0].plot(Adult_Viral_Data['DPI'],trimmed_adult_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 0].plot(Plottable_Adult_Viral_Data['DPI'],Plottable_Adult_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    
    #axs[0,0].plot(t[:-103],trimmed_adult_model3_viral_predictions[:-103],label="M2 Prediction",color="#990000")
    axs[0,0].plot(AdultA1_Virus_times, adult_modelA1_predictions[AdultA1_valid_indices,0][0],label="MA1 Prediction",color=MA1_COLOR)
    axs[0,0].plot(AdultA2_Virus_times, adult_modelA2_predictions[AdultA2_valid_indices,0][0],label="MA2 Prediction",color=MA2_COLOR)
    axs[0,0].plot(AdultA3_Virus_times, adult_modelA3_predictions[AdultA3_valid_indices,0][0],label="MA3 Prediction",color=MA3_COLOR)
    axs[0,0].plot(AdultA4_Virus_times, adult_modelA4_predictions[AdultA4_valid_indices,0][0],label="MA4 Prediction",color=MA4_COLOR)

    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('Virus PFU/ml',fontsize=12)
    axs[0, 0].set_title('Adult',fontsize=16)
    #axs[0, 0].legend()
    
    #axs[0, 1].plot(Aged_Viral_Data['DPI'],trimmed_aged_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 1].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)

    axs[0,1].plot(AgedA1_Virus_times, aged_modelA1_predictions[AgedA1_valid_indices,0][0],label="MA1 Prediction",color=MA1_COLOR)
    axs[0,1].plot(AgedA2_Virus_times, aged_modelA2_predictions[AgedA2_valid_indices,0][0],label="MA2 Prediction",color=MA2_COLOR)
    axs[0,1].plot(AgedA3_Virus_times, aged_modelA3_predictions[AgedA3_valid_indices,0][0],label="MA3 Prediction",color=MA3_COLOR)
    axs[0,1].plot(AgedA4_Virus_times, aged_modelA4_predictions[AgedA4_valid_indices,0][0],label="MA4 Prediction",color=MA4_COLOR)

    axs[0, 1].set_title('Aged',fontsize=16)
    axs[0, 1].set_yscale('log')
    #axs[0, 1].legend()
    


    #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1,0].plot(t, adult_modelA1_predictions[:-1,1],label="MA1 Prediction",color=MA1_COLOR)
    axs[1,0].plot(t, adult_modelA2_predictions[:-1,1],label="MA2 Prediction",color=MA2_COLOR)
    axs[1,0].plot(t, adult_modelA3_predictions[:-1,1],label="MA3 Prediction",color=MA3_COLOR)
    axs[1,0].plot(t, adult_modelA4_predictions[:-1,1],label="MA4 Prediction",color=MA4_COLOR)

    axs[1, 0].set_ylabel('CD8+ per g/tissue',fontsize=12)
    axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1, 1].plot(t, aged_modelA1_predictions[:-1,1],label="MA1 Prediction",color=MA1_COLOR)
    axs[1, 1].plot(t, aged_modelA2_predictions[:-1,1],label="MA2 Prediction",color=MA2_COLOR)
    axs[1, 1].plot(t, aged_modelA3_predictions[:-1,1],label="MA3 Prediction",color=MA3_COLOR)
    axs[1, 1].plot(t, aged_modelA4_predictions[:-1,1],label="MA4 Prediction",color=MA4_COLOR)

    axs[1, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 1].set_xticks(t_vals)
    
    axs[1, 1].legend()
    
    plt.savefig('ModelFitPlots/ModelsA_plot.png')


    ## GROUP B PLOTTING.

    plt.figure()

    fig, axs = plt.subplots(4, 2,figsize=(13,12),sharex='col',sharey='row')

    #axs[0, 0].plot(Adult_Viral_Data['DPI'],trimmed_adult_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 0].plot(Plottable_Adult_Viral_Data['DPI'],Plottable_Adult_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    
    #axs[0,0].plot(t[:-103],trimmed_adult_model3_viral_predictions[:-103],label="M2 Prediction",color="#990000")
    axs[0,0].plot(AdultB1_Virus_times, adult_modelB1_predictions[AdultB1_valid_indices,2][0],label="MB1 Prediction",color=MB1_COLOR)
    axs[0,0].plot(AdultB2_Virus_times, adult_modelB2_predictions[AdultB2_valid_indices,2][0],label="MB2 Prediction",color=MB2_COLOR)
    axs[0,0].plot(AdultB3_Virus_times, adult_modelB3_predictions[AdultB3_valid_indices,2][0],label="MB3 Prediction",color=MB3_COLOR)
    axs[0,0].plot(AdultB4_Virus_times, adult_modelB4_predictions[AdultB4_valid_indices,2][0],label="MB4 Prediction",color=MB4_COLOR)

    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('Virus PFU/ml',fontsize=12)
    axs[0, 0].set_title('Adult',fontsize=16)
    #axs[0, 0].legend()

    #axs[0, 1].plot(Aged_Viral_Data['DPI'],trimmed_aged_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 1].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)

    axs[0,1].plot(AgedB1_Virus_times, aged_modelB1_predictions[AgedB1_valid_indices,2][0],label="MB1 Prediction",color=MB1_COLOR)
    axs[0,1].plot(AgedB2_Virus_times, aged_modelB2_predictions[AgedB2_valid_indices,2][0],label="MB2 Prediction",color=MB2_COLOR)
    axs[0,1].plot(AgedB3_Virus_times, aged_modelB3_predictions[AgedB3_valid_indices,2][0],label="MB3 Prediction",color=MB3_COLOR)
    axs[0,1].plot(AgedB4_Virus_times, aged_modelB4_predictions[AgedB4_valid_indices,2][0],label="MB4 Prediction",color=MB4_COLOR)

    axs[0, 1].set_title('Aged',fontsize=16)
    axs[0, 1].set_yscale('log')
    #axs[0, 1].legend()
    
    #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1,0].plot(t, adult_modelB1_predictions[:-1,3],label="MB1 Prediction",color=MB1_COLOR)
    axs[1,0].plot(t, adult_modelB2_predictions[:-1,3],label="MB2 Prediction",color=MB2_COLOR)
    axs[1,0].plot(t, adult_modelB3_predictions[:-1,3],label="MB3 Prediction",color=MB3_COLOR)
    axs[1,0].plot(t, adult_modelB4_predictions[:-1,3],label="MB4 Prediction",color=MB4_COLOR)

    axs[1, 0].set_ylabel('CD8+ per g/tissue',fontsize=12)
    #axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1, 1].plot(t, aged_modelB1_predictions[:-1,3],label="MB1 Prediction",color=MB1_COLOR)
    axs[1, 1].plot(t, aged_modelB2_predictions[:-1,3],label="MB2 Prediction",color=MB2_COLOR)
    axs[1, 1].plot(t, aged_modelB3_predictions[:-1,3],label="MB3 Prediction",color=MB3_COLOR)
    axs[1, 1].plot(t, aged_modelB4_predictions[:-1,3],label="MB4 Prediction",color=MB4_COLOR)

    #axs[1, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 1].set_xticks(t_vals)
    
    axs[1, 1].legend()

        #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[2,0].plot(t, adult_modelB1_predictions[:-1,0],label="MB1 Prediction",color=MB1_COLOR)
    axs[2,0].plot(t, adult_modelB2_predictions[:-1,0],label="MB2 Prediction",color=MB2_COLOR)
    axs[2,0].plot(t, adult_modelB3_predictions[:-1,0],label="MB3 Prediction",color=MB3_COLOR)
    axs[2,0].plot(t, adult_modelB4_predictions[:-1,0],label="MB4 Prediction",color=MB4_COLOR)

    axs[2, 0].set_ylabel('Uninfected Cells',fontsize=12)
    #axs[2, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[2, 0].set_yscale('log')
    axs[2, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[2, 1].plot(t, aged_modelB1_predictions[:-1,0],label="MB1 Prediction",color=MB1_COLOR)
    axs[2, 1].plot(t, aged_modelB2_predictions[:-1,0],label="MB2 Prediction",color=MB2_COLOR)
    axs[2, 1].plot(t, aged_modelB3_predictions[:-1,0],label="MB3 Prediction",color=MB3_COLOR)
    axs[2, 1].plot(t, aged_modelB4_predictions[:-1,0],label="MB4 Prediction",color=MB4_COLOR)

    #axs[2, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[2, 1].set_xticks(t_vals)
    
        #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[3,0].plot(t, adult_modelB1_predictions[:-1,1],label="MB1 Prediction",color=MB1_COLOR)
    axs[3,0].plot(t, adult_modelB2_predictions[:-1,1],label="MB2 Prediction",color=MB2_COLOR)
    axs[3,0].plot(t, adult_modelB3_predictions[:-1,1],label="MB3 Prediction",color=MB3_COLOR)
    axs[3,0].plot(t, adult_modelB4_predictions[:-1,1],label="MB4 Prediction",color=MB4_COLOR)

    axs[3, 0].set_ylabel('Infected Cells',fontsize=12)
    axs[3, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[3, 0].set_yscale('log')
    axs[3, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[3, 1].plot(t, aged_modelB1_predictions[:-1,1],label="MB1 Prediction",color=MB1_COLOR)
    axs[3, 1].plot(t, aged_modelB2_predictions[:-1,1],label="MB2 Prediction",color=MB2_COLOR)
    axs[3, 1].plot(t, aged_modelB3_predictions[:-1,1],label="MB3 Prediction",color=MB3_COLOR)
    axs[3, 1].plot(t, aged_modelB4_predictions[:-1,1],label="MB4 Prediction",color=MB4_COLOR)

    axs[3, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[3, 1].set_xticks(t_vals)

    plt.savefig('ModelFitPlots/ModelsB_plot.png')

    ## GROUP C PLOTTING.

    plt.figure()

    fig, axs = plt.subplots(2, 2,figsize=(13,6),sharex='col',sharey='row')

    #axs[0, 0].plot(Adult_Viral_Data['DPI'],trimmed_adult_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 0].plot(Plottable_Adult_Viral_Data['DPI'],Plottable_Adult_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    
    #axs[0,0].plot(t[:-103],trimmed_adult_model3_viral_predictions[:-103],label="M2 Prediction",color="#990000")
    axs[0,0].plot(AdultC1_Virus_times, adult_modelC1_predictions[AdultC1_valid_indices,0][0],label="MC1 Prediction",color=MC1_COLOR)
    axs[0,0].plot(AdultC2_Virus_times, adult_modelC2_predictions[AdultC2_valid_indices,0][0],label="MC2 Prediction",color=MC2_COLOR)
    axs[0,0].plot(AdultC3_Virus_times, adult_modelC3_predictions[AdultC3_valid_indices,0][0],label="MC3 Prediction",color=MC3_COLOR)
    axs[0,0].plot(AdultC4_Virus_times, adult_modelC4_predictions[AdultC4_valid_indices,0][0],label="MC4 Prediction",color=MC4_COLOR)

    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('Virus PFU/ml',fontsize=12)
    axs[0, 0].set_title('Adult',fontsize=16)
    #axs[0, 0].legend()

    #axs[0, 1].plot(Aged_Viral_Data['DPI'],trimmed_aged_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 1].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)

    axs[0,1].plot(AgedC1_Virus_times, aged_modelC1_predictions[AgedC1_valid_indices,0][0],label="MC1 Prediction",color=MC1_COLOR)
    axs[0,1].plot(AgedC2_Virus_times, aged_modelC2_predictions[AgedC2_valid_indices,0][0],label="MC2 Prediction",color=MC2_COLOR)
    axs[0,1].plot(AgedC3_Virus_times, aged_modelC3_predictions[AgedC3_valid_indices,0][0],label="MC3 Prediction",color=MC3_COLOR)
    axs[0,1].plot(AgedC4_Virus_times, aged_modelC4_predictions[AgedC4_valid_indices,0][0],label="MC4 Prediction",color=MC4_COLOR)

    axs[0, 1].set_title('Aged',fontsize=16)
    axs[0, 1].set_yscale('log')
    #axs[0, 1].legend()
    
    #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1,0].plot(t, adult_modelC1_predictions[:-1,1],label="MC1 Prediction",color=MC1_COLOR)
    axs[1,0].plot(t, adult_modelC2_predictions[:-1,1],label="MC2 Prediction",color=MC2_COLOR)
    axs[1,0].plot(t, adult_modelC3_predictions[:-1,1],label="MC3 Prediction",color=MC3_COLOR)
    axs[1,0].plot(t, adult_modelC4_predictions[:-1,1],label="MC4 Prediction",color=MC4_COLOR)

    axs[1, 0].set_ylabel('CD8+ per g/tissue',fontsize=12)
    axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1, 1].plot(t, aged_modelC1_predictions[:-1,1],label="MC1 Prediction",color=MC1_COLOR)
    axs[1, 1].plot(t, aged_modelC2_predictions[:-1,1],label="MC2 Prediction",color=MC2_COLOR)
    axs[1, 1].plot(t, aged_modelC3_predictions[:-1,1],label="MC3 Prediction",color=MC3_COLOR)
    axs[1, 1].plot(t, aged_modelC4_predictions[:-1,1],label="MC4 Prediction",color=MC4_COLOR)

    axs[1, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 1].set_xticks(t_vals)
    
    axs[1, 1].legend()
    
    plt.savefig('ModelFitPlots/ModelsC_plot.png')


    ## GROUP D PLOTTING.

    plt.figure()

    fig, axs = plt.subplots(4, 2,figsize=(13,12),sharex='col',sharey='row')

    #axs[0, 0].plot(Adult_Viral_Data['DPI'],trimmed_adult_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 0].plot(Plottable_Adult_Viral_Data['DPI'],Plottable_Adult_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    
    #axs[0,0].plot(t[:-103],trimmed_adult_model3_viral_predictions[:-103],label="M2 Prediction",color="#990000")
    axs[0,0].plot(AdultB1_Virus_times, adult_modelB1_predictions[AdultB1_valid_indices,2][0],label="MD1 Prediction",color=MD1_COLOR)
    axs[0,0].plot(AdultB2_Virus_times, adult_modelB2_predictions[AdultB2_valid_indices,2][0],label="MD2 Prediction",color=MD2_COLOR)
    axs[0,0].plot(AdultB3_Virus_times, adult_modelB3_predictions[AdultB3_valid_indices,2][0],label="MD3 Prediction",color=MD3_COLOR)
    axs[0,0].plot(AdultB4_Virus_times, adult_modelB4_predictions[AdultB4_valid_indices,2][0],label="MD4 Prediction",color=MD4_COLOR)

    axs[0, 0].set_yscale('log')
    axs[0, 0].set_ylabel('Virus PFU/ml',fontsize=12)
    axs[0, 0].set_title('Adult',fontsize=16)
    #axs[0, 0].legend()

    #axs[0, 1].plot(Aged_Viral_Data['DPI'],trimmed_aged_virus,marker='o',linestyle="None",color="#336699")
    axs[0, 1].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)

    axs[0,1].plot(AgedD1_Virus_times, aged_modelD1_predictions[AgedD1_valid_indices,2][0],label="MD1 Prediction",color=MD1_COLOR)
    axs[0,1].plot(AgedD2_Virus_times, aged_modelD2_predictions[AgedD2_valid_indices,2][0],label="MD2 Prediction",color=MD2_COLOR)
    axs[0,1].plot(AgedD3_Virus_times, aged_modelD3_predictions[AgedD3_valid_indices,2][0],label="MD3 Prediction",color=MD3_COLOR)
    axs[0,1].plot(AgedD4_Virus_times, aged_modelD4_predictions[AgedD4_valid_indices,2][0],label="MD4 Prediction",color=MD4_COLOR)

    axs[0, 1].set_title('Aged',fontsize=16)
    axs[0, 1].set_yscale('log')
    #axs[0, 1].legend()
    
    #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1,0].plot(t, adult_modelD1_predictions[:-1,3],label="MD1 Prediction",color=MD1_COLOR)
    axs[1,0].plot(t, adult_modelD2_predictions[:-1,3],label="MD2 Prediction",color=MD2_COLOR)
    axs[1,0].plot(t, adult_modelD3_predictions[:-1,3],label="MD3 Prediction",color=MD3_COLOR)
    axs[1,0].plot(t, adult_modelD4_predictions[:-1,3],label="MD4 Prediction",color=MD4_COLOR)

    axs[1, 0].set_ylabel('CD8+ per g/tissue',fontsize=12)
    #axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1, 1].plot(t, aged_modelD1_predictions[:-1,3],label="MD1 Prediction",color=MD1_COLOR)
    axs[1, 1].plot(t, aged_modelD2_predictions[:-1,3],label="MD2 Prediction",color=MD2_COLOR)
    axs[1, 1].plot(t, aged_modelD3_predictions[:-1,3],label="MD3 Prediction",color=MD3_COLOR)
    axs[1, 1].plot(t, aged_modelD4_predictions[:-1,3],label="MD4 Prediction",color=MD4_COLOR)

    #axs[1, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[1, 1].set_xticks(t_vals)
    
    axs[1, 1].legend()

        #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[2,0].plot(t, adult_modelD1_predictions[:-1,0],label="MD1 Prediction",color=MD1_COLOR)
    axs[2,0].plot(t, adult_modelD2_predictions[:-1,0],label="MD2 Prediction",color=MD2_COLOR)
    axs[2,0].plot(t, adult_modelD3_predictions[:-1,0],label="MD3 Prediction",color=MD3_COLOR)
    axs[2,0].plot(t, adult_modelD4_predictions[:-1,0],label="MD4 Prediction",color=MD4_COLOR)

    axs[2, 0].set_ylabel('Uninfected Cells',fontsize=12)
    #axs[2, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[2, 0].set_yscale('log')
    axs[2, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[2, 1].plot(t, aged_modelD1_predictions[:-1,0],label="MD1 Prediction",color=MD1_COLOR)
    axs[2, 1].plot(t, aged_modelD2_predictions[:-1,0],label="MD2 Prediction",color=MD2_COLOR)
    axs[2, 1].plot(t, aged_modelD3_predictions[:-1,0],label="MD3 Prediction",color=MD3_COLOR)
    axs[2, 1].plot(t, aged_modelD4_predictions[:-1,0],label="MD4 Prediction",color=MD4_COLOR)

    #axs[2, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[2, 1].set_xticks(t_vals)
    
        #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[3,0].plot(t, adult_modelD1_predictions[:-1,1],label="MD1 Prediction",color=MD1_COLOR)
    axs[3,0].plot(t, adult_modelD2_predictions[:-1,1],label="MD2 Prediction",color=MD2_COLOR)
    axs[3,0].plot(t, adult_modelD3_predictions[:-1,1],label="MD3 Prediction",color=MD3_COLOR)
    axs[3,0].plot(t, adult_modelD4_predictions[:-1,1],label="MD4 Prediction",color=MD4_COLOR)

    axs[3, 0].set_ylabel('Infected Cells',fontsize=12)
    axs[3, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[3, 0].set_yscale('log')
    axs[3, 0].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    #axs[1, 1].plot(Aged_CD8['DPI'],Aged_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 1].plot(t,aged_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[3, 1].plot(t, aged_modelD1_predictions[:-1,1],label="MD1 Prediction",color=MD1_COLOR)
    axs[3, 1].plot(t, aged_modelD2_predictions[:-1,1],label="MD2 Prediction",color=MD2_COLOR)
    axs[3, 1].plot(t, aged_modelD3_predictions[:-1,1],label="MD3 Prediction",color=MD3_COLOR)
    axs[3, 1].plot(t, aged_modelD4_predictions[:-1,1],label="MD4 Prediction",color=MD4_COLOR)

    axs[3, 1].set_xlabel('Days Post Infection',fontsize=12)
    axs[3, 1].set_xticks(t_vals)

    plt.savefig('ModelFitPlots/ModelsD_plot.png')