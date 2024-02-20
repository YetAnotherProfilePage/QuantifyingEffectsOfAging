import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt
import math

from GroupColors import *

Fitted_Folder = "LikelihoodProfileFits/"
Destination_Folder = "ModelProfileLikelihoodPlots/"

ModelA1_Fitted_Params_Indices = [0,2,3,4,5]
ModelA1_Params_Labels = ['p','k_V','c_V','r','k_T','d_T']

ModelA2_Fitted_Params_Indices = [0,2,3,4,5,6]
ModelA2_Params_Labels = ['p','k_V','c_V','r','k_T','d_T','c_T']

ModelA3_Fitted_Params_Indices = [0,2,3,4]
ModelA3_Params_Labels = ['p','k_V','c_V','r','d_T']

ModelA4_Fitted_Params_Indices = [0,2,3,4,5]
ModelA4_Params_Labels = ['p','k_V','c_V','r','d_T','c_T']

ModelB1_Fitted_Params_Indices = [0,1,3,4,5,6]
ModelB1_Params_Labels = ['beta','d_I','p','c','d_T','r','k_T']

ModelB2_Fitted_Params_Indices = [0,1,3,4,5,6,7]
ModelB2_Params_Labels = ['beta','d_I','p','c','d_T','r','k_T','c_T']

ModelB3_Fitted_Params_Indices = [0,1,3,4,5]
ModelB3_Params_Labels = ['beta','d_I','p','c','d_T','r']

ModelB4_Fitted_Params_Indices = [0,1,3,4,5,6]
ModelB4_Params_Labels = ['beta','d_I','p','c','d_T','r','c_T']

ModelC1_Fitted_Params_Indices = [0,2,3,4,5,6]
ModelC1_Params_Labels = ['p','k_V','c_V','r','k_T','d_T','K']

ModelC2_Fitted_Params_Indices = [0,2,3,4,5,6,7]
ModelC2_Params_Labels = ['p','k_V','c_V','r','k_T','d_T','c_T','K']

ModelC3_Fitted_Params_Indices = [0,2,3,4,5]
ModelC3_Params_Labels = ['p','k_V','c_V','r','d_T','K']

ModelC4_Fitted_Params_Indices = [0,2,3,4,5,6]
ModelC4_Params_Labels = ['p','k_V','c_V','r','d_T','c_T','K']

ModelD1_Fitted_Params_Indices = [0,1,3,4,5,6,7]
ModelD1_Params_Labels = ['beta','d_I','p','c','d_T','r','k_T','K']

ModelD2_Fitted_Params_Indices = [0,1,3,4,5,6,7,8]
ModelD2_Params_Labels = ['beta','d_I','p','c','d_T','r','k_T','c_T','K']

ModelD3_Fitted_Params_Indices = [0,1,3,4,5,6]
ModelD3_Params_Labels = ['beta','d_I','p','c','d_T','r','K']

ModelD4_Fitted_Params_Indices = [0,1,3,4,5,6,7]
ModelD4_Params_Labels = ['beta','d_I','p','c','d_T','r','c_T','K']

def MakeLikelihoodProfilePlots(model_name,plottitle,label_indices, params_labels, model_color = 'black', fits_folder = Fitted_Folder, save_folder = Destination_Folder):
    active_params = [params_labels[i] for i in label_indices]
    print(active_params)
    num_params = len(active_params)
    print(num_params)

    aged_dfs = []
    adult_dfs = []
    for i in range(num_params):
        aged_temp = pd.read_csv(Fitted_Folder+model_name+"_Aged-"+active_params[i]+".csv")[['RMSLE',active_params[i]]]
        adult_temp = pd.read_csv(Fitted_Folder+model_name+"_Adult-"+active_params[i]+".csv")[['RMSLE',active_params[i]]]
        aged_dfs.append(aged_temp)
        adult_dfs.append(adult_temp)

    ## Now we make the plots
    plt.figure()
    fig, axs = plt.subplots(num_params, 2,figsize=(13,num_params*4.25),sharey='row')
    

    for i in range(num_params):
        if i==0:
            axs[i,0].set_title("Adult",fontsize=16)
            axs[i,1].set_title("Aged",fontsize=16)

        axs[i,0].plot(adult_dfs[i][active_params[i]],adult_dfs[i]['RMSLE'],marker='o',color=model_color)
        axs[i,0].set_xscale('log')
        axs[i,0].set_ylabel('RMSLE',fontsize=12)
        axs[i,0].set_xlabel(f'${active_params[i]}$',fontsize=12)

        axs[i,1].plot(aged_dfs[i][active_params[i]],aged_dfs[i]['RMSLE'],marker='o',color=model_color)
        axs[i,1].set_xscale('log')
        axs[i,1].set_xlabel(f'${active_params[i]}$',fontsize=12)

    fig.suptitle(plottitle,fontsize=18)#,y=)
    fig.subplots_adjust(top=0.95)


    plt.savefig(f'{save_folder}{model_name}_LikelihoodProfilePlots.png')

def MakeMainLikelihoodProfilePlots(model_name,group,plottitle,label_indices, params_labels, model_color = 'black', fits_folder = Fitted_Folder, save_folder = Destination_Folder):
    active_params = [params_labels[i] for i in label_indices]
    print(active_params)
    num_params = len(active_params)
    print(num_params)
    N_COLS = 2

    temp_dfs = []
    for i in range(num_params):
        temp = pd.read_csv(Fitted_Folder+model_name+"_"+group+"-"+active_params[i]+".csv")[['RMSLE',active_params[i]]]
        temp_dfs.append(temp)

    ## Now we make the plots
    plt.figure()
    fig, axs = plt.subplots(math.ceil(num_params/N_COLS), N_COLS,figsize=(13,math.ceil(num_params/2)*4.25),sharey='row')
    

    for i in range(num_params):
        print(f'i: {i}')
        column = i%N_COLS
        row = math.floor(i/N_COLS)
        print(f'column: {column}, row: {row}, column x row: {(column+1)*(row+1)}')
        print(f'column: {column}, row: {row}, {(column+1)+(row*N_COLS)}')

        if column == 0:
            axs[row,column].set_ylabel('RMSLE',fontsize=12)

        axs[row,column].plot(temp_dfs[i][active_params[i]],temp_dfs[i]['RMSLE'],marker='o',color=model_color)
        axs[row,column].set_xscale('log')
        #axs[row,column].set_ylabel('RMSLE',fontsize=12)
        if active_params[i] == "beta":
            axs[row,column].set_xlabel(f'$\\{active_params[i]}$',fontsize=12)
        else:
            axs[row,column].set_xlabel(f'${active_params[i]}$',fontsize=12)

    for i in range(math.ceil(num_params/N_COLS)*N_COLS):
        column = i%N_COLS
        row = math.floor(i/N_COLS)
        
        if (column+1)*(row+1) > num_params:
            print("Here")
            axs[row,column].axis("off")

        
        #axs[i,0].plot(adult_dfs[i][active_params[i]],adult_dfs[i]['RMSLE'],marker='o',color=model_color)
        #axs[i,0].set_xscale('log')
        #axs[i,0].set_ylabel('RMSLE',fontsize=12)
        #axs[i,0].set_xlabel(f'${active_params[i]}$',fontsize=12)

        #axs[i,1].plot(aged_dfs[i][active_params[i]],aged_dfs[i]['RMSLE'],marker='o',color=model_color)
        #axs[i,1].set_xscale('log')
        #axs[i,1].set_xlabel(f'${active_params[i]}$',fontsize=12)

    fig.suptitle(plottitle,fontsize=18)#,y=)
    fig.subplots_adjust(top=0.95)


    plt.savefig(f'{save_folder}{model_name}_{group}_MainLikelihoodProfilePlots.png')


if __name__ == "__main__":
    """
    MakeLikelihoodProfilePlots("ModelA1","MA1 Likelihood Profiles",ModelA1_Fitted_Params_Indices,ModelA1_Params_Labels,model_color=MA1_COLOR)
    MakeLikelihoodProfilePlots("ModelA2","MA2 Likelihood Profiles",ModelA2_Fitted_Params_Indices,ModelA2_Params_Labels,model_color=MA2_COLOR)
    MakeLikelihoodProfilePlots("ModelA3","MA3 Likelihood Profiles",ModelA3_Fitted_Params_Indices,ModelA3_Params_Labels,model_color=MA3_COLOR)
    MakeLikelihoodProfilePlots("ModelA4","MA4 Likelihood Profiles",ModelA4_Fitted_Params_Indices,ModelA4_Params_Labels,model_color=MA4_COLOR)

    MakeLikelihoodProfilePlots("ModelB1","MB1 Likelihood Profiles",ModelB1_Fitted_Params_Indices,ModelB1_Params_Labels,model_color=MB1_COLOR)
    MakeLikelihoodProfilePlots("ModelB2","MB2 Likelihood Profiles",ModelB2_Fitted_Params_Indices,ModelB2_Params_Labels,model_color=MB2_COLOR)
    MakeLikelihoodProfilePlots("ModelB3","MB3 Likelihood Profiles",ModelB3_Fitted_Params_Indices,ModelB3_Params_Labels,model_color=MB3_COLOR)
    MakeLikelihoodProfilePlots("ModelB4","MB4 Likelihood Profiles",ModelB4_Fitted_Params_Indices,ModelB4_Params_Labels,model_color=MB4_COLOR)

    MakeLikelihoodProfilePlots("ModelC1","MC1 Likelihood Profiles",ModelC1_Fitted_Params_Indices,ModelC1_Params_Labels,model_color=MC1_COLOR)
    MakeLikelihoodProfilePlots("ModelC2","MC2 Likelihood Profiles",ModelC2_Fitted_Params_Indices,ModelC2_Params_Labels,model_color=MC2_COLOR)
    MakeLikelihoodProfilePlots("ModelC3","MC3 Likelihood Profiles",ModelC3_Fitted_Params_Indices,ModelC3_Params_Labels,model_color=MC3_COLOR)
    MakeLikelihoodProfilePlots("ModelC4","MC4 Likelihood Profiles",ModelC4_Fitted_Params_Indices,ModelC4_Params_Labels,model_color=MC4_COLOR)

    MakeLikelihoodProfilePlots("ModelD1","MD1 Likelihood Profiles",ModelD1_Fitted_Params_Indices,ModelD1_Params_Labels,model_color=MD1_COLOR)
    MakeLikelihoodProfilePlots("ModelD2","MD2 Likelihood Profiles",ModelD2_Fitted_Params_Indices,ModelD2_Params_Labels,model_color=MD2_COLOR)
    MakeLikelihoodProfilePlots("ModelD3","MD3 Likelihood Profiles",ModelD3_Fitted_Params_Indices,ModelD3_Params_Labels,model_color=MD3_COLOR)
    MakeLikelihoodProfilePlots("ModelD4","MD4 Likelihood Profiles",ModelD4_Fitted_Params_Indices,ModelD4_Params_Labels,model_color=MD4_COLOR)
    """

    MakeMainLikelihoodProfilePlots("ModelC1","Adult","Adult Likelihood Profiles (MC1)",ModelC1_Fitted_Params_Indices,ModelC1_Params_Labels,model_color=MC1_COLOR)
    MakeMainLikelihoodProfilePlots("ModelD1","Aged","Aged Likelihood Profiles (MD1)",ModelD1_Fitted_Params_Indices,ModelD1_Params_Labels,model_color=MD1_COLOR)

