import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

SOURCE_FOLDER = "GlobalSensitivityAnalysisResults/"
DESTINATION_FOLDER = "GlobalSensitivityAnalysisPlots/"

# For now, these plots will only be for Adult.
Models = ["MA1","MA2","MA3","MA4","MB1","MB2","MB3","MB4","MC1","MC2","MC3","MC4","MD1","MD2","MD3","MD4"]
#Age_Groups = ["Adult","Aged"]
Age_Groups = ["Adult"]

Outcomes = ["ExcessCTL","TotalViralLoad","ViralClearanceTime"]
#Indices = ["First","Second","Total"]
Indices= ["First","Total"]
PrettyOutcomes = ["Excess CTL", "Total Viral Load", "Viral Clearance Time"]

# We will ultimately use the combinations to iterate over all sensitivity files.

combinations = list(itertools.product(Age_Groups,Models))

def LaTeXifyParameters(params):
    new_params = []

    for param in params:
        temp_p = param.replace("'","").replace("(","").replace(")","").replace(" ","").split(",")

        #If I were to extend this, I would want to handle special characters in broader generality.

        temp_p2 = ["\\beta" if p=="beta" else p for p in temp_p]

        if len(temp_p2)>1:
            temp = [f"${i}$" for i in temp_p2]

            temp_p2 = "-".join(temp)
        else:
            temp_p2 = f"${param}$"

        new_params.append(temp_p2)

    return new_params
    #"""

def MakeModelGlobalSensitivityPlot(model,group):

    dfs = []
    for index in Indices:
        outcome_dfs = []
        for outcome in Outcomes:            
            outcome_dfs.append(pd.read_csv(SOURCE_FOLDER+f"{group}_{model}_{outcome}_{index}_Indices.csv"))
        dfs.append(outcome_dfs)

    # each row will be an index, 
    fig, axs = plt.subplots(len(Indices),1, figsize=(13,12))

    for i in range(len(Indices)):
        curr_dfs = dfs[i]

        params = list(curr_dfs[0]["Unnamed: 0"])

        #each parameter is a group, each outcome puts a bar in that group.


        width = 0.25  # the width of the bars
        multiplier = 0 
        x = np.arange(len(params))

        # The columns are named different for first, second, and total order indices.
        if i==0:
            key = "S1"
        elif i==1:
            key = "S2"
            continue
        else:
            key = "ST"

        for o in range(len(Outcomes)):
            offset = width*multiplier
            #rects = axs[i].bar(x+offset, curr_dfs[o][f"{key}"],width,yerr=curr_dfs[o][f"{key}_conf"],label=Outcomes[o])
            rects = axs[i].bar(x+offset, curr_dfs[o][f"{key}"],width,label=PrettyOutcomes[o])
            #axs[i].bar_label(rects,padding =3)
            multiplier += 1
        
        axs[i].set_ylabel(f"{Indices[i]} Order",fontsize=14)
        #ax.set_title('Penguin attributes by species')
        axs[i].set_xticks(x + width, LaTeXifyParameters(params))
        
        print()

    axs[0].legend(loc='upper right', ncols=3,fontsize=12)
    axs[0].set_title(f"Model {model} Sobol Sensitivity Indices",fontsize=16)

    plt.savefig(DESTINATION_FOLDER+f"{model}_{group}_GlobalSensitivityPlot.png")

    #print()

def MakeModelGlobalSensitivityPlot_OnlyFirstAndTotal(model,group):

    dfs = []
    for index in Indices:
        outcome_dfs = []
        for outcome in Outcomes:            
            outcome_dfs.append(pd.read_csv(SOURCE_FOLDER+f"{group}_{model}_{outcome}_{index}_Indices.csv"))
        dfs.append(outcome_dfs)

    # each row will be an index, 
    fig, axs = plt.subplots(len(Indices),1, figsize=(13,12))

    for i in range(len(Indices)):
        curr_dfs = dfs[i]

        params = list(curr_dfs[0]["Unnamed: 0"])

        #each parameter is a group, each outcome puts a bar in that group.


        width = 0.25  # the width of the bars
        multiplier = 0 
        x = np.arange(len(params))

        # The columns are named different for first, second, and total order indices.
        if i==0:
            key = "S1"
        else:
            key = "ST"

        for o in range(len(Outcomes)):
            offset = width*multiplier
            #rects = axs[i].bar(x+offset, curr_dfs[o][f"{key}"],width,yerr=curr_dfs[o][f"{key}_conf"],label=Outcomes[o])
            rects = axs[i].bar(x+offset, curr_dfs[o][f"{key}"],width,label=PrettyOutcomes[o])
            #axs[i].bar_label(rects,padding =3)
            multiplier += 1
        
        axs[i].set_ylabel(f"{Indices[i]} Order",fontsize=14)
        axs[i].set_yscale('log')
        axs[i].set_xticks(x + width, LaTeXifyParameters(params))
        
        print()

    axs[0].legend(loc='upper right', ncols=3,fontsize=12)
    axs[0].set_title(f"Model {model} Sobol Sensitivity Indices",fontsize=16)

    plt.savefig(DESTINATION_FOLDER+f"{model}_{group}_GlobalSensitivityPlot.png")

    #print()

if __name__ == "__main__":
    print(LaTeXifyParameters(["('beta','s_T')",]))
    MakeModelGlobalSensitivityPlot_OnlyFirstAndTotal("MC1","Adult")
    MakeModelGlobalSensitivityPlot_OnlyFirstAndTotal("MD1","Adult")