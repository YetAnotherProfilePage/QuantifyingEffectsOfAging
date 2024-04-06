import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt

import math

from GroupColors import *

Fitted_Folder = "BootstrappingFits/"
Destination_Folder = "ModelBootstrappingPlots/"

## If I had thought more carefully about it, I would have structured these dicts in a seperate file to begin with.
Models_Params_Labels = {
    "MA1":['p','k_V','c_V','r','k_T','d_T'],
    "MA2":['p','k_V','c_V','r','k_T','d_T','c_T'],
    "MA3":['p','k_V','c_V','r','d_T'],
    "MA4":['p','k_V','c_V','r','d_T','c_T'],
    "MB1":['beta','d_I','p','c','d_T','r','k_T'],
    "MB2":['beta','d_I','p','c','d_T','r','k_T','c_T'],
    "MB3":['beta','d_I','p','c','d_T','r'],
    "MB4":['beta','d_I','p','c','d_T','r','c_T'],
    "MC1":['p','k_V','c_V','r','k_T','d_T','K'],
    "MC2":['p','k_V','c_V','r','k_T','d_T','c_T','K'],
    "MC3":['p','k_V','c_V','r','d_T','K'],
    "MC4":['p','k_V','c_V','r','d_T','c_T','K'],
    "MD1":['beta','d_I','p','c','d_T','r','k_T','K'],
    "MD2":['beta','d_I','p','c','d_T','r','k_T','c_T','K'],
    "MD3":['beta','d_I','p','c','d_T','r','K'],
    "MD4":['beta','d_I','p','c','d_T','r','c_T','K'],
}

Models_Params_Indices = {
    "MA1":[0,2,3,4,5],
    "MA2":[0,2,3,4,5,6],
    "MA3":[0,2,3,4],
    "MA4":[0,2,3,4,5],
    "MB1":[0,1,3,4,5,6],
    "MB2":[0,1,3,4,5,6,7],
    "MB3":[0,1,3,4,5],
    "MB4":[0,1,3,4,5,6],
    "MC1":[0,2,3,4,5,6],
    "MC2":[0,2,3,4,5,6,7],
    "MC3":[0,2,3,4,5],
    "MC4":[0,2,3,4,5,6],
    "MD1":[0,1,3,4,5,6,7],
    "MD2":[0,1,3,4,5,6,7,8],
    "MD3":[0,1,3,4,5,6],
    "MD4":[0,1,3,4,5,6,7],
}

import itertools

#ModelGroupLabels = ["MA","MB","MC","MD"]
#ModelMechLabels = ["1","2","3","4"]
ModelGroupLabels=["MC","MD"]
ModelMechLabels=["1"]
DataGroups = ["Adult","Aged"]

test =itertools.product(ModelGroupLabels,ModelMechLabels,DataGroups)

file_names = []
for i in test:
    file_name = i[0]+i[1]+"_"+i[2]+"_Bootstrapping.csv"
    print(file_name)
    file_names.append(file_name)

print(file_names)

def find_exp(number) -> int:
    base10 = math.log10(number)
    return math.floor(base10)

def MakeBootstrappingPlot(data_file, model_name, data_group,label_indices, params_labels, model_color = 'black', fits_folder = Fitted_Folder, save_folder = Destination_Folder,ALPHA = 0.5,plot_title=None):
    #print(label_indices)
    #print(params_labels)
    print(model_name)
    active_params = [params_labels[i] for i in label_indices]
    print(active_params)
    num_params = len(active_params)
    print(num_params)

    rows = num_params
    cols = num_params

    df = pd.read_csv(fits_folder+data_file)
    print(df)

    print(f"Number of params = {num_params}")
    print(f"Active parameters = {active_params}")

    plt.figure()
    #BASIC_DIMS = 2.75
    BASIC_DIMS = 2.25
    fig, axs = plt.subplots(num_params-1, num_params-1,figsize=((num_params-1)*BASIC_DIMS,(num_params-1)*BASIC_DIMS),sharey='row',sharex='col')

    all_axes = list(itertools.product(list(range(num_params-1)),list(range(num_params-1))))

    for i in range(rows):
        for j in range(cols):
            #print(f'({i},{j})')
            if j < i:
                #print(f'({i},{j+1})')
                #print(f'({active_params[i]},{active_params[j]})')

                axs[i-1,j].plot(df[active_params[j]],df[active_params[i]],marker='o',linestyle='',color=model_color,alpha=ALPHA)
                #axs[i-1,j].set_xlabel(f"${active_params[j]}$")
                #axs[i-1,j].set_ylabel(f"${active_params[i]}$")
                if i == num_params-1:
                    if active_params[j] == "beta":
                        axs[i-1,j].set_xlabel(f"$\\{active_params[j]}$")
                    else:
                        axs[i-1,j].set_xlabel(f"${active_params[j]}$")
                if j == 0:
                    if active_params[i] == "beta":
                        axs[i-1,j].set_ylabel(f"$\\{active_params[i]}$")
                    else:
                        axs[i-1,j].set_ylabel(f"${active_params[i]}$")

                axs[i-1,j].loglog()

                """

                if active_params[j]=="beta":
                    #print("Here")
                    min_x = min(df[active_params[j]])
                    max_x = max(df[active_params[j]])
                    min_exp = find_exp(min_x)
                    max_exp = find_exp(max_x)
                    #print(find_exp(min_x))
                    #print(find_exp(max_x))
                    #print(f'Min: {min_x},Max: {max_x}')
                    axs[i-1,j].set_xticks([10**min_exp,10**(max_exp+1)],[f'$10^{{{min_exp}}}$',f'$10^{{{max_exp+1}}}$'])
                """
                min_x = min(df[active_params[j]])
                max_x = max(df[active_params[j]])
                min_exp = find_exp(min_x)
                max_exp = find_exp(max_x)
                axs[i-1,j].set_xticks([],[],minor=True)
                axs[i-1,j].set_xticks([10**min_exp,10**(max_exp+1)],[f'$10^{{{min_exp}}}$',f'$10^{{{max_exp+1}}}$'])

                #axs[i-1,j].set_xscale('log')
                #axs[i-1,j].set_xticklabels(axs[i-1,j].get_xticks(), rotation = 90)
                #axs[i-1,j].ticklabel_format(style="plain")
                #for tick in axs[i-1,j].get_xticklabels():
                #    tick.set_rotation(45)

                #axs[i-1,j].set_yscale('log')

                all_axes.remove((i-1,j))
            else:
                pass
                #print(f"({i},{j})")
                #if i == num_params-2 and j == num_params-2:
                #    axs[i,j].axis('off')

    for idx in all_axes:
        axs[idx[0],idx[1]].axis('off')

    if plot_title==None:
        plot_title = f"{model_name} {data_group} Bootstrapped Parameters Scatterplot"

    fig.suptitle(plot_title,fontsize=18)#,y=)
    fig.subplots_adjust(top=0.95)

    plt.savefig(f'{save_folder}{model_name}_{data_group}_BootstrappingPlots.png')

    print(all_axes)
    """

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


    plt.savefig(f'{save_folder}{model_name}_BootstrappingPlots.png')
    """

"""
plt.figure(figsize=(16,12))

fig, axs = plt.subplots(3, 3,figsize=(13,6),sharex='col',sharey='row')
fig.suptitle('Adult',fontsize=16)

axs[0,0].plot(adult[['p']],adult[['c_V']],'o',color="blue",alpha=ALPHA)
axs[0,0].set_ylabel('$c_V$', fontsize=12)

axs[0,1].axis('off')
axs[0,2].axis('off')

axs[1,0].plot(adult[['p']],adult[['k_T']],'o',color="blue",alpha=ALPHA)
axs[1,0].set_yscale('log')
axs[1,0].set_ylabel('$k_T$', fontsize=12)

axs[2,0].plot(adult[['p']],adult[['c_T']],'o',color="blue",alpha=ALPHA)
axs[2,0].set_ylabel('$c_T$', fontsize=12)
axs[2,0].set_xlabel('$p$',fontsize=12)
axs[2,0].set_yscale('log')

axs[1,1].plot(adult[['c_V']],adult[['k_T']],'o',color="blue",alpha=ALPHA)

axs[1,2].axis("off")

axs[2,1].plot(adult[['c_V']],adult[['c_T']],'o',color="blue",alpha=ALPHA)
axs[2,1].set_xlabel('$c_V$',fontsize=12)

axs[2,2].plot(adult[['k_T']],adult[['c_T']],'o',color="blue",alpha=ALPHA)
axs[2,2].set_xlabel('$k_T$',fontsize=12)
axs[2,2].set_xscale('log')

fig.align_ylabels(axs[:,0])

plt.savefig('images/adult_bootstrapping.png',dpi=400)
"""

if __name__ == "__main__":
    for file_name in file_names:
        temp = file_name.split("_")
        mod = temp[0]
        group = temp[1]
        print(mod)
        print(group)
        print(file_name)
        MakeBootstrappingPlot(file_name,mod,group,Models_Params_Indices[mod],Models_Params_Labels[mod],model_color=COLORS_DICT[mod])

    """
    print(file_names[0])
    print(file_names[0][0:3])
    #print(file_names[0][4:10])
    print(file_names[0].split("_"))
    temp = file_names[0].split("_")
    mod = temp[0]
    group = temp[1]
    print(mod)
    print(group)
    MakeBootstrappingPlot(file_names[0],mod,group,Models_Params_Indices[mod],Models_Params_Labels[mod],model_color=COLORS_DICT[mod])
    """