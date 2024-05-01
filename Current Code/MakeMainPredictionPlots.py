import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from GroupColors import *

from Models.ModelA3 import *

from Models.ModelB1 import *


VIRAL_THRESHOLD = 24.0

CD8_Data = pd.read_csv("ExperimentalData/ReorganizedData_CD8.csv")
Viral_Data = pd.read_csv("ExperimentalData/ReorganizedData_ViralTiter.csv")

#TODO: Make it so that replacement occurs for DPI > 5
Viral_Data['Viral Titer (Pfu/ml)'] = Viral_Data['Viral Titer (Pfu/ml)'].replace(0.0,1e-8).copy()
Viral_Data.loc[Viral_Data['DPI']==0,'Viral Titer (Pfu/ml)'] = 25.0

Adult_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Adult']
Aged_Viral_Data = Viral_Data[Viral_Data['Age Group']=='Aged']

Adult_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Adult']
Aged_CD8_Data = CD8_Data[CD8_Data['Age Group'] == 'Aged']

ALPHA = 0.4
t = np.linspace(0,19,190)

#t_vals is used for plotting
#t_vals = [i for i in range(1,20)]
t_vals = [i for i in range(0,20,2)]
### Load model fits

## Adult Group (MC1)

adult_modelA3_params = np.loadtxt("ModelFits/adult_modelA3_params.txt")

adult_modelA3_predictions = ModelA3_Predict(adult_modelA3_params, 19,10)

## Aged Group (MD1)

aged_modelB1_params = np.loadtxt("ModelFits/aged_modelB1_params.txt")

aged_modelB1_predictions = ModelB1_Predict(aged_modelB1_params, 19,10)


## Clip virus:

AdultA3_valid_indices = np.where(adult_modelA3_predictions[:,0]>VIRAL_THRESHOLD)
Adult_Virus_times = t[AdultA3_valid_indices]

AgedB1_valid_indices = np.where(aged_modelB1_predictions[:,2]>VIRAL_THRESHOLD)
Aged_Virus_times = t[AgedB1_valid_indices]


print(Aged_Virus_times)
print(Aged_Virus_times[-1])
print(aged_modelB1_predictions[AgedB1_valid_indices,2][0])

# Clip data:
# Adult data is only valid through day 9

Plottable_Adult_Viral_Data = Adult_Viral_Data[Adult_Viral_Data["DPI"]<=9]

# Aged data is only valid through day 11
Plottable_Aged_Viral_Data = Aged_Viral_Data[Aged_Viral_Data["DPI"]<=11]

LINEWIDTH = 2.25

if __name__ == "__main__":
        
    ## GROUP A PLOTTING.

    plt.figure()

    fig, axs = plt.subplots(2, 1,figsize=(13,6),sharex='col',sharey='row')

    axs[0].plot(Plottable_Adult_Viral_Data['DPI'],Plottable_Adult_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="red",alpha=ALPHA)
    
    #axs[0].plot(t, adult_modelC1_predictions[:-1,0],label="MC1 Prediction",color=MC1_COLOR)
    axs[0].plot(Adult_Virus_times, adult_modelA3_predictions[AdultA3_valid_indices,0][0],label="MA3 Prediction",color="blue",alpha=0.5,linewidth=LINEWIDTH)
    axs[0].axhline(y=25.0, color='grey', linestyle='dotted')


    axs[0].set_yscale('log')
    axs[0].set_ylabel('Virus (PFU/ml)',fontsize=12)
    axs[0].set_title('Adult Model Predictions',fontsize=16)
    
    axs[1].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    axs[1].plot(t, adult_modelA3_predictions[:-1,1],label="MA3 Prediction",color="blue",alpha=0.5,linewidth=LINEWIDTH)
    
    axs[1].set_ylabel('CD8+ T Cells',fontsize=12)
    axs[1].set_xlabel('Days Post Infection',fontsize=12)
    axs[1].set_yscale('log')
    axs[1].set_xticks(t_vals)
    #axs[1, 0].legend()
    
    
    plt.savefig('ModelFitPlots/MainAdultPlot.png')


    # Plot for aged model: D1.
    plt.figure()

    fig, axs = plt.subplots(4, 1,figsize=(13,12),sharex='col',sharey='row')

    #axs[0, 0].plot(Adult_Viral_Data['DPI'],trimmed_adult_virus,marker='o',linestyle="None",color="#336699")
    print(axs.shape)

    axs[0].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="red",alpha=ALPHA)
    #axs[0].plot(t, aged_modelD1_predictions[:-1,2],label="MD1 Prediction",color=MD1_COLOR)
    axs[0].plot(Aged_Virus_times, aged_modelB1_predictions[AgedB1_valid_indices,2][0],label="MB1 Prediction",color=MB1_COLOR)
    axs[0].axhline(y=25.0, color='grey', linestyle='dotted')
    
    axs[0].set_yscale('log')
    axs[0].set_ylabel('Virus (PFU/ml)',fontsize=12)
    axs[0].set_title('Aged Model Predictions',fontsize=16)

    #axs[0, 0].legend()

    #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1].plot(t, aged_modelB1_predictions[:-1,3],label="MB1 Prediction",color=MB1_COLOR)
    
    axs[1].set_ylabel('CD8+ T Cells',fontsize=12)
    #axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1].set_yscale('log')
    axs[1].set_xticks(t_vals)
    axs[2].plot(t, aged_modelB1_predictions[:-1,0],label="MB1 Prediction",color=MB1_COLOR)
    
    axs[2].set_ylabel('Uninfected Cells',fontsize=12)
    #axs[2, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[2].set_yscale('log')
    axs[2].set_xticks(t_vals)
    
    axs[3].plot(t, aged_modelB1_predictions[:-1,1],label="MB1 Prediction",color=MB1_COLOR)
    
    axs[3].set_ylabel('Infected Cells',fontsize=12)
    axs[3].set_xlabel('Days Post Infection',fontsize=12)
    axs[3].set_yscale('symlog')
    axs[3].set_xticks(t_vals)
    
    plt.savefig('ModelFitPlots/MainAgedPlot.png')


    ## Now aged and adult side by side.

    plt.figure()

    fig, axs = plt.subplots(2, 2,figsize=(13,6),sharex='col',sharey='row')

    axs[0,0].plot(Plottable_Adult_Viral_Data['DPI'],Plottable_Adult_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="red",alpha=ALPHA)
    
    #axs[0].plot(t, adult_modelC1_predictions[:-1,0],label="MC1 Prediction",color=MC1_COLOR)
    #axs[0,0].plot(Adult_Virus_times, adult_modelC1_predictions[AdultC1_valid_indices,0][0],label="MC1 Prediction",color=MC1_COLOR)
    axs[0,0].plot(Adult_Virus_times, adult_modelA3_predictions[AdultA3_valid_indices,0][0],label="MA3 Prediction",color="blue",alpha=0.5,linewidth=LINEWIDTH)
    axs[0,0].axhline(y=25.0, color='grey', linestyle='dotted')

    axs[0,0].set_yscale('log')
    axs[0,0].set_ylabel('Virus (PFU/ml)',fontsize=12)
    axs[0,0].set_title('Adult Model Trajectories',fontsize=16)
    
    axs[1,0].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="red",alpha=ALPHA)
    #axs[1,0].plot(t, adult_modelC1_predictions[:-1,1],label="MC1 Prediction",color=MC1_COLOR)
    axs[1,0].plot(t, adult_modelA3_predictions[:-1,1],label="MA3 Prediction",color="blue",alpha=0.5,linewidth=LINEWIDTH)
    
    axs[1,0].set_ylabel('CD8+ T Cells',fontsize=12)
    axs[1,0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1,0].set_yscale('log')
    axs[1,0].set_xticks(t_vals)
    #axs[1, 0].legend()

    axs[0,1].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="red",alpha=ALPHA)
    #axs[0].plot(t, aged_modelD1_predictions[:-1,2],label="MD1 Prediction",color=MD1_COLOR)
    #axs[0,1].plot(Aged_Virus_times, aged_modelD1_predictions[AgedD1_valid_indices,2][0],label="MD1 Prediction",color=MD1_COLOR)
    axs[0,1].plot(Aged_Virus_times, aged_modelB1_predictions[AgedB1_valid_indices,2][0],label="MB1 Prediction",color="darkorange",linewidth=LINEWIDTH)
    axs[0,1].axhline(y=25.0, color='grey', linestyle='dotted')
    
    axs[0,1].set_yscale('log')
    #axs[0,1].set_ylabel('Virus (PFU/ml)',fontsize=12)
    axs[0,1].set_title('Aged Model Trajectories',fontsize=16)

    #axs[0, 0].legend()

    #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    axs[1,1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="red",alpha=ALPHA)
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    #axs[1,1].plot(t, aged_modelD1_predictions[:-1,3],label="MD1 Prediction",color=MD1_COLOR)
    axs[1,1].plot(t, aged_modelB1_predictions[:-1,3],label="MB1 Prediction",color="darkorange",linewidth=LINEWIDTH)
    
    #axs[1,1].set_ylabel('CD8+ T Cells',fontsize=12)
    #axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1,1].set_yscale('log')
    axs[1,1].set_xticks(t_vals)
    axs[1,1].set_xlabel('Days Post Infection',fontsize=12)


    #fig.suptitle("Predictions from the Selected Models",fontsize=16)

    plt.savefig('ModelFitPlots/MainPredictionPlot.png')

    ## Now just the trajectories overlaid.


    plt.figure()

    fig, axs = plt.subplots(2, 1,figsize=(8,6),sharex='col',sharey='row')

    #axs[0].plot(t, adult_modelC1_predictions[:-1,0],label="MC1 Prediction",color=MC1_COLOR)
    axs[0].plot(Adult_Virus_times, adult_modelA3_predictions[AdultA3_valid_indices,0][0],label="Adult (MA3)",color="blue",alpha=0.5,linewidth=LINEWIDTH)
    axs[0].axhline(y=25.0, color='grey', linestyle='dotted')
    #axs[0].axhline(y=25.0, color='grey', linestyle='dotted')

    axs[0].set_yscale('log')
    axs[0].set_ylabel('Virus (PFU/ml)',fontsize=12)
    axs[0].set_title('Model Trajectories',fontsize=16)
    
    axs[1].plot(t, adult_modelA3_predictions[:-1,1],label="Adult (MA3)",color="blue",alpha=0.5,linewidth=LINEWIDTH)
    
    axs[1].set_ylabel('CD8+ T Cells',fontsize=12)
    axs[1].set_xlabel('Days Post Infection',fontsize=12)
    axs[1].axhline(y=aged_modelB1_params[-1],linestyle='dotted',color="blue",alpha=0.5,label="Aged (MB1)")
    axs[1].set_yscale('log')
    axs[1].set_xticks(t_vals)

    #axs[0].plot(t, aged_modelD1_predictions[:-1,2],label="MD1 Prediction",color=MD1_COLOR)
    axs[0].plot(Aged_Virus_times, aged_modelB1_predictions[AgedB1_valid_indices,2][0],label="Aged (MB1)",color="darkorange",linewidth=LINEWIDTH)

    #axs[0, 0].legend()

    #axs[1, 0].plot(Adult_CD8['DPI'],Adult_CD8['CD8+ per g/tissue'],marker='o',linestyle="None",color="#336699")
    #axs[1, 0].plot(t,adult_model3_cd8_predictions,label='M2 Prediction',color="#990000")
    axs[1].plot(t, aged_modelB1_predictions[:-1,3],label="Aged (MB1)",color="darkorange",linewidth=LINEWIDTH)
    axs[1].axhline(y=adult_modelA3_params[-2],linestyle='dotted',color="darkorange",label="Adult (MA3)")
    
    #axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
    axs[1].set_xlabel('Days Post Infection',fontsize=12)

    axs[0].legend()

    #fig.suptitle("Predictions from the Selected Models",fontsize=16)

    plt.savefig('ModelFitPlots/MainPredictionPlot_NoData.png')

    print(aged_modelB1_params)
    print(adult_modelA3_params)

    print(aged_modelB1_params[-1])
    print(adult_modelA3_params[-2])
