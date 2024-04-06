import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

import math
import itertools

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

T_0_Adult = np.mean(Adult_CD8_Data[Adult_CD8_Data['DPI']==0]['CD8+ per g/tissue'])
T_0_Aged = np.mean(Aged_CD8_Data[Aged_CD8_Data['DPI']==0]['CD8+ per g/tissue'])

t = np.linspace(0,19,190)

#Note: We do not distinguish k_V between Aged and Adult.
k_V = np.max(Viral_Data['Viral Titer (Pfu/ml)']) #k_V = 1.2e6


Fitted_Folder = "BootstrappingFits/"
Destination_Folder = "BootstrappedPredictionsPlots/"

ModelGroupLabels = ["MA","MB","MC","MD"]
ModelMechLabels = ["1","2","3","4"]
DataGroups = ["Adult","Aged"]

Models_Inputs = {
    "MA1":['p','k_V','c_V','r','k_T','d_T','T_0','V_0'],
    "MA2":['p','k_V','c_V','r','k_T','d_T','c_T','T_0','V_0'],
    "MA3":['p','k_V','c_V','r','d_T','T_0','V_0'],
    "MA4":['p','k_V','c_V','r','d_T','c_T','T_0','V_0'],
    "MB1":['beta','d_I','p','c','d_T','r','k_T','U_0', 'I_0', 'V_0', 'T_0'],
    "MB2":['beta','d_I','p','c','d_T','r','k_T','c_T','U_0', 'I_0', 'V_0', 'T_0'],
    "MB3":['beta','d_I','p','c','d_T','r','U_0', 'I_0', 'V_0', 'T_0'],
    "MB4":['beta','d_I','p','c','d_T','r','c_T','U_0', 'I_0', 'V_0', 'T_0'],
    "MC1":['p','k_V','c_V','r','k_T','d_T','K','T_0','V_0'],
    "MC2":['p','k_V','c_V','r','k_T','d_T','c_T','K','T_0','V_0'],
    "MC3":['p','k_V','c_V','r','d_T','K','T_0','V_0'],
    "MC4":['p','k_V','c_V','r','d_T','c_T','K','T_0','V_0'],
    "MD1":['beta','d_I','p','c','d_T','r','k_T','K','U_0', 'I_0', 'V_0', 'T_0'],
    "MD2":['beta','d_I','p','c','d_T','r','k_T','c_T','K','U_0', 'I_0', 'V_0', 'T_0'],
    "MD3":['beta','d_I','p','c','d_T','r','K','U_0', 'I_0', 'V_0', 'T_0'],
    "MD4":['beta','d_I','p','c','d_T','r','c_T','K','U_0', 'I_0', 'V_0', 'T_0'],
}

Models_Dict = {
    "MA1":ModelA1_Predict,
    "MA2":ModelA2_Predict,
    "MA3":ModelA3_Predict,
    "MA4":ModelA4_Predict,
    "MB1":ModelB1_Predict,
    "MB2":ModelB2_Predict,
    "MB3":ModelB3_Predict,
    "MB4":ModelB4_Predict,
    "MC1":ModelC1_Predict,
    "MC2":ModelC2_Predict,
    "MC3":ModelC3_Predict,
    "MC4":ModelC4_Predict,
    "MD1":ModelD1_Predict,
    "MD2":ModelD2_Predict,
    "MD3":ModelD3_Predict,
    "MD4":ModelD4_Predict,
}

test =itertools.product(ModelGroupLabels,ModelMechLabels,DataGroups)

file_names = []
for i in test:
    file_name = i[0]+i[1]+"_"+i[2]+"_Bootstrapping.csv"
    print(file_name)
    file_names.append(file_name)

print(file_names)

# Clip data:
# Adult data is only valid through day 9

Plottable_Adult_Viral_Data = Adult_Viral_Data[Adult_Viral_Data["DPI"]<=9]

# Aged data is only valid through day 11
Plottable_Aged_Viral_Data = Aged_Viral_Data[Aged_Viral_Data["DPI"]<=11]

ALPHA = 0.5
t = np.linspace(0,19,190)
t_vals = [i for i in range(0,20,2)]

VIRAL_THRESHOLD = 24.0


if __name__ == "__main__":

    file_names= ["MC1_Adult_Bootstrapping.csv","MD1_Aged_Bootstrapping.csv"]

    for i in file_names:
        #print(i)
        #print(i[0:3])
        desc = i.split("_")
        print(desc)
        model = i[0:3]

        df = pd.read_csv(Fitted_Folder+i)[Models_Inputs[i[0:3]]]
        params = df.to_numpy()
        #print(params)
        #print(params.shape[0])

        # We break the plotting into cases

        if model[0:2] in ['MA','MC']:
            print(2)

            #Make predictions
            predictions = []
            for idx in range(params.shape[0]):
                pred = Models_Dict[model](params[idx,:],19,10)
                predictions.append(pred)

            #print(params.size())
            
            plt.figure()

            fig, axs = plt.subplots(2, 1,figsize=(13,6),sharex='col',sharey='row')

            if desc[1]=="Adult":
                axs[0].plot(Plottable_Adult_Viral_Data['DPI'],Plottable_Adult_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
                axs[1].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
            
            else:
                axs[0].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
                axs[1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)

            
            #axs[0].plot(t, adult_modelC1_predictions[:-1,0],label="MC1 Prediction",color=MC1_COLOR)
            #axs[0].plot(Adult_Virus_times, adult_modelC1_predictions[AdultC1_valid_indices,0][0],label="MC1 Prediction",color=MC1_COLOR)
            axs[0].axhline(y=25.0, color='grey', linestyle='dotted')


            axs[0].set_yscale('log')
            axs[0].set_ylabel('Virus (PFU/ml)',fontsize=12)
            
            print(desc[1])

            axs[0].set_title(f'{desc[0]} {desc[1]} Bootstrapped Model Predictions',fontsize=16)
            
            #axs[1].plot(t, adult_modelC1_predictions[:-1,1],label="MC1 Prediction",color=MC1_COLOR)
            
            axs[1].set_ylabel('CD8+ T Cells',fontsize=12)
            axs[1].set_xlabel('Days Post Infection',fontsize=12)
            axs[1].set_yscale('log')
            axs[1].set_xticks(t_vals)


            for current in predictions:
                ## Clip virus:
                #current = predictions[i]
                valid_indices = np.where(current[:,0]>VIRAL_THRESHOLD)
                virus_times = t[valid_indices]

                axs[0].plot(virus_times, current[valid_indices,0][0],color=COLORS_DICT[model],alpha=0.1)
                axs[1].plot(t, current[:-1,1],color=COLORS_DICT[model],alpha=0.1)

            plt.savefig(f'{Destination_Folder}{desc[0]}_{desc[1]}_BootstrappingPredictions.png')

            
        elif model[0:2] in ['MB','MD']:

            predictions = []
            for idx in range(params.shape[0]):
                pred = Models_Dict[model](params[idx,:],19,10)
                predictions.append(pred)


            plt.figure()

            fig, axs = plt.subplots(4, 1,figsize=(13,12),sharex='col',sharey='row')

            #axs[0, 0].plot(Adult_Viral_Data['DPI'],trimmed_adult_virus,marker='o',linestyle="None",color="#336699")
            print(axs.shape)

            if desc[1]=="Adult":
                axs[0].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
                axs[1].plot(Adult_CD8_Data['DPI'],Adult_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)

            else:
                axs[0].plot(Plottable_Aged_Viral_Data['DPI'],Plottable_Aged_Viral_Data['Viral Titer (Pfu/ml)'],marker='o',linestyle="None",color="grey",alpha=ALPHA)
                axs[1].plot(Aged_CD8_Data['DPI'],Aged_CD8_Data['CD8+ per g/tissue'],marker='o',linestyle="None",color="grey",alpha=ALPHA)


            #axs[0].plot(t, aged_modelD1_predictions[:-1,2],label="MD1 Prediction",color=MD1_COLOR)
            axs[0].axhline(y=25.0, color='grey', linestyle='dotted')
            
            axs[0].set_yscale('log')
            axs[0].set_ylabel('Virus (PFU/ml)',fontsize=12)
            axs[0].set_title(f'{desc[0]} {desc[1]} Bootstrapped Model Predictions',fontsize=16)

            
            axs[1].set_ylabel('CD8+ T Cells',fontsize=12)
            #axs[1, 0].set_xlabel('Days Post Infection',fontsize=12)
            axs[1].set_yscale('log')
            axs[1].set_xticks(t_vals)
            
            axs[2].set_ylabel('Uninfected Cells',fontsize=12)
            #axs[2, 0].set_xlabel('Days Post Infection',fontsize=12)
            axs[2].set_yscale('log')
            axs[2].set_xticks(t_vals)
                        
            axs[3].set_ylabel('Infected Cells',fontsize=12)
            axs[3].set_xlabel('Days Post Infection',fontsize=12)
            axs[3].set_yscale('symlog')
            axs[3].set_xticks(t_vals)

            for current in predictions:
                ## Clip virus:
                #current = predictions[i]
                valid_indices = np.where(current[:,2]>VIRAL_THRESHOLD)
                virus_times = t[valid_indices]

                axs[0].plot(virus_times, current[valid_indices,2][0],color=COLORS_DICT[model],alpha=0.1)
                axs[1].plot(t, current[:-1,3],color=COLORS_DICT[model],alpha=0.1)
                axs[2].plot(t, current[:-1,0],color=COLORS_DICT[model],alpha=0.1)
                axs[3].plot(t, current[:-1,1],color=COLORS_DICT[model],alpha=0.1)
                
            plt.savefig(f'{Destination_Folder}{desc[0]}_{desc[1]}_BootstrappingPredictions.png')
            print(4)
        else:
            print(-1)
            print("Something is wrong.")


