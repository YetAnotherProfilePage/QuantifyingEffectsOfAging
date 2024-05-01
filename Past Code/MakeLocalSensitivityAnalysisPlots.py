import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

from os import walk

from GroupColors import *
from ModelBounds import *
from FittingUtilities import *

MODEL_FITS_FOLDER = "ModelFits/"
SOURCE_DIRECTORY = "LocalSensitivityAnalysisResults/"
DESTINATION_DIRECTORY = "LocalSensitivityAnalysisPlots/"

# We want to mimic profile likelihood plots for each model group and outcome.

# Get list of files in folder

OUTCOMES = ["Viral Clearance Time", "Total Viral Load", "Excess CTL"]
FIG_SCALE = 4.0

def Get_Files_In_Dir(DIR):
    f = []
    for (dirpath, dirnames, filenames) in walk(SOURCE_DIRECTORY):
        f.extend(filenames)
        break
    return f

def Get_Model_Group_Files(model, group, filenames):
    selected_filenames = []
    for i in filenames:
        temp = i.split("-")
        if (temp[0]==model and temp[1]==group):
            selected_filenames.append(i)
    return selected_filenames

def Make_Model_Group_Local_Sensitivity_Plot(model,group,filenames):
    files = Get_Model_Group_Files(model,group,filenames)
    #print(files)
    params = [i[2] for i in [j.split("-") for j in files]]
    print(params)
    num_params = len(params)
    dfs = [pd.read_csv(SOURCE_DIRECTORY+i) for i in files]
    #print(dfs)

    fig, axs = plt.subplots(len(OUTCOMES),num_params,figsize=((FIG_SCALE)*len(OUTCOMES),FIG_SCALE/4*num_params),sharey="row",sharex="col")

    for i in range(0,num_params):
        for j in range(0, len(OUTCOMES)):
            axs[j,i].plot(dfs[i][params[i]],dfs[i][OUTCOMES[j]],color=COLORS_DICT[model])
            axs[j,i].set_xscale("log")
            axs[j,i].tick_params(axis='both',which="both",labelsize=8)

            if OUTCOMES[j]!="Viral Clearance Time":
                axs[j,i].set_yscale("log")

            if j == (len(OUTCOMES)-1):
                if params[i] =="beta":
                    axs[j,i].set_xlabel(f"$\\{params[i]}$")
                else:
                    axs[j,i].set_xlabel(f"${params[i]}$")
            if i == 0:
                axs[j,i].set_ylabel(OUTCOMES[j])

    fig.suptitle(f"Model {model} {group} fit local sensitivity",fontsize=16)

    print("Here")

    plt.savefig(DESTINATION_DIRECTORY+f"{model}_{group}_LocalSensitivityPlot.png")

if __name__ == "__main__":

    filenames = Get_Files_In_Dir(SOURCE_DIRECTORY)
    print(filenames)

    print(Get_Model_Group_Files("MA1","aged",filenames))

    Make_Model_Group_Local_Sensitivity_Plot("MC1","adult",filenames)
    Make_Model_Group_Local_Sensitivity_Plot("MD1","aged",filenames)



